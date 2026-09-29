"""
textlens.anpr.tracking
──────────────────────
Plate tracking across video frames with temporal voting.

A single frame is often blurred or partially occluded.  Tracking plates
(IoU association) and voting per character across frames — weighted by
per-character confidence — turns several mediocre reads into one reliable
plate, which matters far more in practice than any single-frame model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from textlens.anpr.types import PlateRead


@dataclass
class Track:
    id: int
    bbox: object
    last_frame: int
    reads: List[PlateRead] = field(default_factory=list)

    def vote(self) -> Optional[PlateRead]:
        """Confidence-weighted per-position vote over reads of the modal length."""
        if not self.reads:
            return None
        by_len: Dict[int, float] = defaultdict(float)
        for r in self.reads:
            by_len[len(r.plate)] += r.confidence
        length = max(by_len, key=by_len.get)
        cands = [r for r in self.reads if len(r.plate) == length]
        chars = []
        char_conf = []
        for i in range(length):
            scores: Dict[str, float] = defaultdict(float)
            for r in cands:
                c = r.char_confidences[i] if i < len(r.char_confidences) else r.confidence
                scores[r.plate[i]] += c * (1.5 if r.valid else 1.0)
            best = max(scores, key=scores.get)
            chars.append(best)
            char_conf.append(scores[best] / max(1e-9, sum(scores.values())))
        best_read = max(cands, key=lambda r: r.confidence)
        plate = "".join(chars)
        agreement = sum(char_conf) / len(char_conf) if char_conf else 0.0
        return PlateRead(
            plate=plate,
            confidence=round(min(1.0, 0.5 * agreement + 0.5 * max(r.confidence for r in cands)), 4),
            bbox=best_read.bbox,
            raw_text=best_read.raw_text,
            detection_confidence=best_read.detection_confidence,
            recognition_confidence=best_read.recognition_confidence,
            char_confidences=char_conf,
            valid=any(r.valid and r.plate == plate for r in cands) or best_read.valid,
            format=best_read.format,
            region=best_read.region,
            vehicle_type=best_read.vehicle_type,
            track_id=self.id,
            detector=best_read.detector,
            recognizer=best_read.recognizer,
        )


class PlateTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, max_reads: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_reads = max_reads
        self.tracks: Dict[int, Track] = {}
        self._next = 1

    def update(self, frame: int, reads: List[PlateRead]) -> List[PlateRead]:
        """Assign track ids and return per-track voted readings for this frame."""
        assigned = set()
        out = []
        for read in sorted(reads, key=lambda r: r.confidence, reverse=True):
            best_id, best_iou = None, self.iou_threshold
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                iou = read.bbox.iou(tr.bbox)  # type: ignore[arg-type]
                if iou >= best_iou:
                    best_id, best_iou = tid, iou
            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = Track(best_id, read.bbox, frame)
            tr = self.tracks[best_id]
            tr.bbox, tr.last_frame = read.bbox, frame
            read.track_id = best_id
            tr.reads.append(read)
            tr.reads = tr.reads[-self.max_reads:]
            assigned.add(best_id)
            voted = tr.vote()
            if voted is not None:
                out.append(voted)
        for tid in [t for t, tr in self.tracks.items() if frame - tr.last_frame > self.max_age]:
            del self.tracks[tid]
        return out
