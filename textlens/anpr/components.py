"""
textlens.anpr.components
────────────────────────
Pluggable ANPR stages.  Each stage is a small protocol so detectors and
recognisers can be swapped (or replaced by your own trained models) without
touching the pipeline.

Detectors
    ``YoloPlateDetector``     open-image-models YOLOv9 plate detectors (ONNX, MIT)
    ``TextRegionPlateDetector`` zero-extra-dependency fallback: PP-OCR text
                               regions filtered by plate geometry
    ``FullImageDetector``     the input is already a plate crop

Recognisers
    ``FastPlateRecognizer``   fast-plate-ocr CCT / MobileViT models trained on
                               plates (per-character confidence, region head)
    ``TextLensRecognizer``    any TextLens OCR model on the crop (fallback)

Vehicle detector (optional)
    ``CocoVehicleDetector``   RF-DETR COCO models → car / truck / bus / motorcycle
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from textlens.core.result import BBox
from textlens.errors import BackendUnavailableError

logger = logging.getLogger("textlens.anpr")

_ANPR_HINT = 'pip install "textlens-ocr[anpr]"'


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    label: str = "plate"
    polygon: Optional[List[Tuple[float, float]]] = None


@dataclass
class Recognition:
    text: str
    confidence: float
    char_confidences: List[float] = field(default_factory=list)
    region: Optional[str] = None


class PlateDetector(Protocol):
    name: str

    def detect(self, image: Any) -> List[Detection]: ...


class PlateRecognizer(Protocol):
    name: str

    def recognize(self, crops: Sequence[Any]) -> List[Recognition]: ...


def _ort_providers(device: Optional[str]) -> Optional[List[Any]]:
    if device is None:
        return None
    from textlens.backends.onnx.runtime import select_providers

    return select_providers(device)


# ── detectors ────────────────────────────────────────────────────────────────


class YoloPlateDetector:
    """open-image-models YOLOv9 end-to-end plate detector."""

    MODELS = {
        "edge": "yolo-v9-t-384-license-plate-end2end",
        "balanced": "yolo-v9-t-640-license-plate-end2end",
        "accurate": "yolo-v9-s-608-license-plate-end2end",
    }

    def __init__(self, model: str = "yolo-v9-t-640-license-plate-end2end", conf_thresh: float = 0.4, device: Optional[str] = None) -> None:
        try:
            from open_image_models import create_detector
        except ImportError as exc:
            raise BackendUnavailableError("Plate detection needs open-image-models.", hint=_ANPR_HINT) from exc
        self.name = model
        self._det = create_detector(model, conf_thresh=conf_thresh, providers=_ort_providers(device))

    def detect(self, image: Any) -> List[Detection]:
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        out = []
        for r in self._det.predict(bgr):
            b = r.bounding_box
            out.append(Detection(BBox(float(b.x1), float(b.y1), float(b.x2), float(b.y2)), float(r.confidence), r.label))
        return out


class CocoVehicleDetector:
    """Vehicle boxes from an RF-DETR COCO model (for ``vehicle_type``)."""

    VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}

    def __init__(self, model: str = "rf-detr-nano-384-coco", conf_thresh: float = 0.4, device: Optional[str] = None) -> None:
        try:
            from open_image_models import create_detector
        except ImportError as exc:
            raise BackendUnavailableError("Vehicle detection needs open-image-models.", hint=_ANPR_HINT) from exc
        self.name = model
        self._det = create_detector(model, conf_thresh=conf_thresh, providers=_ort_providers(device))

    def detect(self, image: Any) -> List[Detection]:
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        out = []
        for r in self._det.predict(bgr):
            if r.label in self.VEHICLES:
                b = r.bounding_box
                out.append(Detection(BBox(float(b.x1), float(b.y1), float(b.x2), float(b.y2)), float(r.confidence), r.label))
        return out


class TextRegionPlateDetector:
    """Fallback detector: PP-OCR text regions that look like plates.

    Needs no extra packages.  Plate-shaped regions (aspect 1.5–7, 4–10
    alphanumerics) are kept.  Accuracy is below a trained plate detector —
    use it for pre-cropped or clearly visible plates.
    """

    name = "textlens-text-regions"

    def __init__(self, device: Optional[str] = None, profile: str = "edge") -> None:
        from textlens.core.engine import OCR

        self._ocr = OCR(model="ppocrv6-small", device=device, cache="off")

    def detect(self, image: Any) -> List[Detection]:
        from textlens.anpr.validation import normalize

        result = self._ocr(image)
        out = []
        for line in result.lines:
            if line.bbox is None:
                continue
            aspect = line.bbox.width / max(1.0, line.bbox.height)
            n = len(normalize(line.text))
            if 1.5 <= aspect <= 7.5 and 4 <= n <= 10:
                pad = 0.15 * line.bbox.height
                b = BBox(max(0, line.bbox.x0 - pad), max(0, line.bbox.y0 - pad), min(image.size[0], line.bbox.x1 + pad), min(image.size[1], line.bbox.y1 + pad))
                out.append(Detection(b, float(line.confidence or 0.5), polygon=line.polygon))
        return out


class FullImageDetector:
    """Treat the whole input as one plate (inputs that are already crops)."""

    name = "full-image"

    def detect(self, image: Any) -> List[Detection]:
        w, h = image.size
        return [Detection(BBox(0, 0, w, h), 1.0)]


# ── recognisers ──────────────────────────────────────────────────────────────


class FastPlateRecognizer:
    """fast-plate-ocr models (trained on license plates)."""

    MODELS = {"edge": "cct-xs-v2-global-model", "balanced": "cct-s-v2-global-model", "accurate": "cct-s-v2-global-model"}

    def __init__(self, model: str = "cct-s-v2-global-model", device: Optional[str] = None) -> None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError as exc:
            raise BackendUnavailableError("Plate recognition needs fast-plate-ocr.", hint=_ANPR_HINT) from exc
        self.name = model
        dev = "cpu" if (device or "").startswith("cpu") else ("cuda" if (device or "").startswith("cuda") else "auto")
        self._rec = LicensePlateRecognizer(model, device=dev)
        mode = getattr(getattr(self._rec, "config", None), "image_color_mode", "grayscale")
        self._rgb = str(mode).lower() == "rgb"

    def recognize(self, crops: Sequence[Any]) -> List[Recognition]:
        if not crops:
            return []
        arrays = [np.asarray(c.convert("RGB" if self._rgb else "L")) for c in crops]
        preds = self._rec.run(arrays, return_confidence=True)
        out = []
        for p in preds:
            probs = [float(x) for x in (p.char_probs if p.char_probs is not None else [])][: len(p.plate)]
            conf = float(np.mean(probs)) if probs else 0.0
            out.append(Recognition(p.plate, conf, probs, getattr(p, "region", None)))
        return out


class TextLensRecognizer:
    """Recognise crops with a TextLens OCR model (PP-OCRv6 by default)."""

    def __init__(self, model: str = "ppocrv6-small", device: Optional[str] = None) -> None:
        from textlens.core.engine import OCR

        self.name = model
        self._ocr = OCR(model=model, device=device, cache="off")

    def recognize(self, crops: Sequence[Any]) -> List[Recognition]:
        out = []
        for crop in crops:
            result = self._ocr(crop)
            lines = [ln for ln in result.lines if ln.text.strip()]
            text = " ".join(ln.text for ln in lines)  # two-row plates are joined
            confs = [ln.confidence for ln in lines if ln.confidence is not None]
            chars: List[float] = []
            for ln in lines:
                for w in ln.words:
                    chars.extend([w.confidence or 0.0] * len(w.text))
            out.append(Recognition(text, float(np.mean(confs)) if confs else 0.0, chars))
        return out


# ── rectification ────────────────────────────────────────────────────────────


def crop_plate(image: Any, det: Detection, margin: float = 0.08, deskew: bool = True, target_height: int = 96) -> Any:
    """Crop a plate with a margin, correct perspective/skew, normalise size."""
    from PIL import Image

    w, h = image.size
    if det.polygon and len(det.polygon) == 4:
        tl, tr, br, bl = [np.array(p, np.float64) for p in det.polygon]
        cw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        ch = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        crop = image.transform((max(cw, 2), max(ch, 2)), Image.QUAD,
                               (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1]), Image.BICUBIC)
    else:
        b = det.bbox
        mx, my = margin * b.width, margin * b.height
        box = (int(max(0, b.x0 - mx)), int(max(0, b.y0 - my)), int(min(w, b.x1 + mx)), int(min(h, b.y1 + my)))
        crop = image.crop(box)
        if deskew and min(crop.size) >= 12:
            angle = estimate_skew(crop)
            if abs(angle) >= 1.0:
                crop = crop.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if crop.size[1] and crop.size[1] < target_height:
        scale = target_height / crop.size[1]
        crop = crop.resize((max(1, int(crop.size[0] * scale)), target_height), Image.BICUBIC)
    return crop


def estimate_skew(crop: Any, max_angle: float = 15.0, step: float = 1.0) -> float:
    """Skew angle (degrees) maximising horizontal projection sharpness."""
    from PIL import Image

    g = crop.convert("L")
    g.thumbnail((200, 200))
    arr = np.asarray(g, np.float32)
    thresh = arr.mean() - 0.5 * arr.std()
    mask = arr < thresh
    if arr.std() < 8 or mask.mean() < 0.01:  # blank or uniform: nothing to align
        return 0.0
    ink = Image.fromarray((mask * 255).astype(np.uint8))

    def score(angle: float) -> float:
        rot = np.asarray(ink.rotate(angle, resample=Image.NEAREST, expand=False), np.float32)
        return float(np.var(rot.sum(axis=1)))

    base = score(0.0)
    best_angle, best_score = 0.0, base
    angle = -max_angle
    while angle <= max_angle + 1e-6:
        if angle != 0.0:
            sc = score(angle)
            if sc > best_score:
                best_score, best_angle = sc, angle
        angle += step
    # Only rotate for a clear improvement; small gains are noise.
    if best_score < base * 1.05 or not math.isfinite(best_angle):
        return 0.0
    return best_angle
