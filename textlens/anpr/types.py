"""Result types for number-plate recognition (pixel coordinates, top-left origin)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from textlens.core.result import BBox


@dataclass
class PlateRead:
    plate: str  # normalised, validated/corrected text (e.g. "TN01AB1234")
    confidence: float  # combined detection × recognition × validation
    bbox: BBox  # plate box in the input image
    raw_text: str = ""  # recogniser output before normalisation/correction
    detection_confidence: Optional[float] = None
    recognition_confidence: Optional[float] = None
    char_confidences: List[float] = field(default_factory=list)
    polygon: Optional[List[Tuple[float, float]]] = None
    valid: bool = False  # matches the configured region format
    format: Optional[str] = None  # name of the matched format rule
    corrections: List[str] = field(default_factory=list)  # e.g. ["pos 3: O→0"]
    region: Optional[str] = None  # recogniser's region prediction, if any
    vehicle_type: Optional[str] = None  # car | truck | bus | motorcycle (vehicle detector)
    vehicle_bbox: Optional[BBox] = None
    track_id: Optional[int] = None
    detector: Optional[str] = None
    recognizer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = [round(v, 1) for v in self.bbox]
        if self.vehicle_bbox is not None:
            d["vehicle_bbox"] = [round(v, 1) for v in self.vehicle_bbox]
        d["confidence"] = round(self.confidence, 4)
        d["char_confidences"] = [round(c, 4) for c in self.char_confidences]
        return d


@dataclass
class ANPRResult:
    plates: List[PlateRead]
    width: int
    height: int
    timings_ms: Dict[str, float] = field(default_factory=dict)
    profile: Optional[str] = None
    frame: Optional[int] = None

    @property
    def best(self) -> Optional[PlateRead]:
        return max(self.plates, key=lambda p: p.confidence, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plates": [p.to_dict() for p in self.plates],
            "width": self.width,
            "height": self.height,
            "timings_ms": self.timings_ms,
            "profile": self.profile,
            "frame": self.frame,
        }

    def annotate(self, image: Any) -> Any:
        """Draw plate boxes and readings onto a copy of ``image``."""
        from PIL import ImageDraw

        from textlens.inputs.images import load_image, to_rgb
        from textlens.inputs.source import open_source

        if not hasattr(image, "size"):
            src = open_source(image)
            image = src.image if src.image is not None else load_image(src.open_binary())
        img = to_rgb(image).copy()
        draw = ImageDraw.Draw(img)
        for p in self.plates:
            color = (46, 204, 113) if p.valid else (241, 196, 15)
            draw.rectangle(list(p.bbox), outline=color, width=max(2, img.size[0] // 400))
            label = f"{p.plate} {p.confidence:.2f}" + (f" #{p.track_id}" if p.track_id is not None else "")
            y = max(0, p.bbox.y0 - 14)
            draw.rectangle([p.bbox.x0, y, p.bbox.x0 + 8 * len(label), y + 13], fill=color)
            draw.text((p.bbox.x0 + 2, y), label, fill=(0, 0, 0))
            if p.vehicle_bbox is not None:
                draw.rectangle(list(p.vehicle_bbox), outline=(52, 152, 219), width=2)
        return img
