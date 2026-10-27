"""
textlens.backends.onnx.ppocr
────────────────────────────
PP-OCRv6 (detection + recognition) on ONNX Runtime — the TextLens edge and
default engine.  Dependencies: numpy, Pillow, onnxruntime.  No OpenCV, no
PyTorch, no Paddle.

Pipeline per image
    1. Detection: resize (longest side ≤ ``det_limit``, multiples of 32),
       ImageNet-normalise (BGR), DBNet probability map, numpy post-process.
    2. Adaptive escalation: if detected text is tiny after downscaling
       (dense pages, small fonts), detection re-runs at a higher resolution.
    3. Crop each quadrilateral with a perspective (quad) warp; vertical
       crops are rotated.
    4. Recognition: height-48 crops batched by aspect ratio, CTC greedy
       decoding over the 18.7k-symbol PP-OCRv6 dictionary.
    5. Per-line confidence = mean max-probability of emitted characters.

Model artifacts are pinned by SHA-256 in the ``ppocrv6-small`` spec and
downloaded (~31 MB) on first use unless offline mode is on.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from textlens.backends.base import OCRBackend, PageOCR, RecognizeOptions
from textlens.backends.onnx.dbnet import boxes_from_probability
from textlens.backends.onnx.runtime import make_session, provider_to_device
from textlens.core.result import BBox, Word
from textlens.documents.layout import Item

logger = logging.getLogger("textlens.onnx.ppocr")

_DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_REC_HEIGHT = 48


class PPOCRBackend(OCRBackend):
    """PP-OCRv6 Small via ONNX Runtime.

    Options
    -------
    det_limit : int
        Longest side fed to the detector (default 1280; edge profile 960).
    det_thresh, box_thresh, unclip_ratio : float
        DBNet post-processing parameters.
    rec_batch : int
        Recognition batch size.
    threads : int
        ONNX Runtime intra-op threads (default: all cores).
    low_memory : bool
        Disable ORT memory arenas (small boards).
    min_line_confidence : float
        Drop recognised lines below this confidence (default 0.3).
    auto_download : bool
        Download artifacts on first use (default True, honours offline mode).
    """

    thread_safe = True
    backend_name = "onnxruntime"

    def _load(self) -> None:
        from textlens.models import artifacts

        opts = self.options
        if not artifacts.is_installed(self.spec):
            if not opts.get("auto_download", True):
                from textlens.errors import ModelNotInstalledError

                raise ModelNotInstalledError(self.spec.id)
            logger.warning("Downloading %s (~%.0f MB, SHA-256 verified)…", self.spec.display_name, (self.spec.download_size_gb or 0) * 1000)
            artifacts.install(self.spec)
        root = artifacts.model_dir(self.spec)
        names = {a.filename for a in self.spec.artifacts}
        det_name = next(n for n in names if "det" in n and n.endswith(".onnx"))
        rec_name = next(n for n in names if "rec" in n and n.endswith(".onnx"))
        dict_name = next(n for n in names if n.endswith(".txt"))
        threads = opts.get("threads") or None
        low_mem = bool(opts.get("low_memory", False))
        self._det, provider = make_session(str(root / det_name), self.requested_device, threads, low_mem)
        self._rec, _ = make_session(str(root / rec_name), self.requested_device, threads, low_mem)
        self.device = provider_to_device(provider)
        with open(root / dict_name, encoding="utf-8") as fh:
            vocab = fh.read().splitlines()
        self._chars = ["<blank>"] + vocab + [" "]
        self._det_input = self._det.get_inputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name

    def _unload(self) -> None:
        self._det = None
        self._rec = None

    # ── detection ────────────────────────────────────────────────────────
    def _detect(self, img: Any, limit: int) -> List[Tuple[np.ndarray, float]]:
        from PIL import Image

        w, h = img.size
        ratio = min(1.0, limit / float(max(w, h)))
        nw = max(32, int(round(w * ratio / 32.0)) * 32)
        nh = max(32, int(round(h * ratio / 32.0)) * 32)
        arr = np.asarray(img.resize((nw, nh), Image.BILINEAR), np.float32)[:, :, ::-1] / 255.0
        arr = ((arr - _DET_MEAN) / _DET_STD).transpose(2, 0, 1)[None]
        prob = self._det.run(None, {self._det_input: np.ascontiguousarray(arr, dtype=np.float32)})[0][0, 0]
        o = self.options
        return boxes_from_probability(
            prob,
            (w / float(nw), h / float(nh)),
            thresh=float(o.get("det_thresh", 0.3)),
            box_thresh=float(o.get("box_thresh", 0.6)),
            unclip_ratio=float(o.get("unclip_ratio", 1.5)),
        )

    # ── recognition ──────────────────────────────────────────────────────
    @staticmethod
    def _crop(img: Any, quad: np.ndarray) -> Any:
        from PIL import Image

        tl, tr, br, bl = quad
        cw = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        ch = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        cw, ch = max(cw, 2), max(ch, 2)
        crop = img.transform(
            (cw, ch),
            Image.QUAD,
            (float(tl[0]), float(tl[1]), float(bl[0]), float(bl[1]), float(br[0]), float(br[1]), float(tr[0]), float(tr[1])),
            Image.BICUBIC,
        )
        if ch >= 1.5 * cw:
            crop = crop.rotate(90, expand=True)
        return crop

    def _recognize_crops(self, crops: Sequence[Any]) -> List[Tuple[str, float, List[float]]]:
        from PIL import Image

        results: List[Optional[Tuple[str, float, List[float]]]] = [None] * len(crops)
        ratios = [c.size[0] / float(max(1, c.size[1])) for c in crops]
        order = np.argsort(ratios)
        batch = int(self.options.get("rec_batch", 8))
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            max_ratio = max(ratios[i] for i in ids)
            width = max(320, int(math.ceil(_REC_HEIGHT * max_ratio)))
            tensor = np.zeros((len(ids), 3, _REC_HEIGHT, width), np.float32)
            for k, i in enumerate(ids):
                nw = min(width, max(1, int(math.ceil(_REC_HEIGHT * ratios[i]))))
                arr = np.asarray(crops[i].resize((nw, _REC_HEIGHT), Image.BILINEAR), np.float32)[:, :, ::-1] / 255.0
                tensor[k, :, :, :nw] = ((arr - 0.5) / 0.5).transpose(2, 0, 1)
            logits = self._rec.run(None, {self._rec_input: tensor})[0]
            for k, i in enumerate(ids):
                idx = logits[k].argmax(1)
                prob = logits[k].max(1)
                keep = (idx != 0) & np.concatenate([[True], idx[1:] != idx[:-1]])
                chars = [self._chars[j] if j < len(self._chars) else "" for j in idx[keep]]
                char_probs = [float(p) for p in prob[keep]]
                text = "".join(chars).strip()
                conf = float(np.mean(prob[keep])) if keep.any() else 0.0
                results[i] = (text, conf, char_probs)
        return [r if r is not None else ("", 0.0, []) for r in results]

    # ── public entry ─────────────────────────────────────────────────────
    def _recognize(self, images: List[Any], options: RecognizeOptions) -> List[PageOCR]:
        out: List[PageOCR] = []
        limit = int(self.options.get("det_limit", 1280))
        max_limit = int(self.options.get("det_max_limit", 2560))
        min_conf = float(self.options.get("min_line_confidence", 0.3))
        for img in images:
            img = img.convert("RGB") if img.mode != "RGB" else img
            t0 = time.perf_counter()
            boxes = self._detect(img, limit)
            # Escalate when text is too small for the detector at this scale.
            if boxes and max(img.size) > limit and limit < max_limit:
                heights = sorted(min(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[1] - q[0])) for q, _ in boxes)
                median_h = heights[len(heights) // 2] * (limit / float(max(img.size)))
                if median_h < 12.0:
                    boxes = self._detect(img, min(max_limit, int(limit * 2)))
            t1 = time.perf_counter()
            crops = [self._crop(img, q) for q, _ in boxes]
            recs = self._recognize_crops(crops) if crops else []
            t2 = time.perf_counter()
            items: List[Item] = []
            confs: List[Tuple[float, int]] = []
            for (quad, det_score), (text, conf, char_probs) in zip(boxes, recs):
                if not text or conf < min_conf:
                    continue
                bbox = BBox.from_points(quad).rounded(1)
                polygon = [(round(float(x), 1), round(float(y), 1)) for x, y in quad]
                words = _split_words(text, bbox, char_probs)
                items.append(Item(text=text, bbox=bbox, confidence=round(conf, 4), words=words, polygon=polygon))
                confs.append((conf, len(text)))
            total = sum(n for _, n in confs)
            page_conf = round(sum(c * n for c, n in confs) / total, 4) if total else None
            out.append(
                PageOCR(
                    width=img.size[0],
                    height=img.size[1],
                    items=items,
                    confidence=page_conf,
                    timings_ms={"detect": round((t1 - t0) * 1000, 1), "recognize": round((t2 - t1) * 1000, 1)},
                )
            )
        return out


def _split_words(text: str, bbox: BBox, char_probs: List[float]) -> List[Word]:
    """Approximate word boxes by character position along the line box."""
    tokens = text.split(" ")
    if len(tokens) <= 1:
        conf = round(float(np.mean(char_probs)), 4) if char_probs else None
        return [Word(text, bbox, conf)]
    n = max(1, len(text))
    words: List[Word] = []
    pos = 0
    for tok in tokens:
        if tok:
            x0 = bbox.x0 + bbox.width * pos / n
            x1 = bbox.x0 + bbox.width * (pos + len(tok)) / n
            probs = char_probs[pos : pos + len(tok)] if char_probs else []
            words.append(Word(tok, BBox(x0, bbox.y0, x1, bbox.y1).rounded(1), round(float(np.mean(probs)), 4) if probs else None))
        pos += len(tok) + 1
    return words
