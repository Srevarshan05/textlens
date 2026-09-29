"""
textlens.anpr.pipeline
──────────────────────
Number-plate recognition (ANPR / LPR) as a dedicated pipeline — not generic
OCR on a whole photo::

    image ─► vehicle detection (optional) ─► plate detection ─► crop + deskew/perspective
          ─► plate recognition ─► region-format validation & correction ─► confidence
          ─► (video) tracking + temporal voting ─► ANPRResult

    >>> from textlens import ANPR
    >>> anpr = ANPR(profile="edge", region="in")
    >>> result = anpr("vehicle.jpg")
    >>> [p.plate for p in result.plates]

Profiles pick detector/recogniser sizes:

=========  ===================================  ==========================
profile    detector (open-image-models)         recogniser (fast-plate-ocr)
=========  ===================================  ==========================
edge       yolo-v9-t-384                        cct-xs-v2-global
balanced   yolo-v9-t-640                        cct-s-v2-global
accurate   yolo-v9-s-608                        cct-s-v2-global
=========  ===================================  ==========================

Without the ``anpr`` extra the pipeline falls back to TextLens' own PP-OCR
(text regions filtered by plate geometry) so it still runs, at lower
accuracy.  All stages are injectable for custom models.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Iterator, List, Optional

from textlens.anpr.components import (
    CocoVehicleDetector,
    FastPlateRecognizer,
    FullImageDetector,
    TextLensRecognizer,
    TextRegionPlateDetector,
    YoloPlateDetector,
    crop_plate,
)
from textlens.anpr.formats import Region, load_region
from textlens.anpr.tracking import PlateTracker
from textlens.anpr.types import ANPRResult, PlateRead
from textlens.anpr.validation import PlateValidator
from textlens.errors import BackendUnavailableError, ConfigurationError

logger = logging.getLogger("textlens.anpr")

ANPR_PROFILES = ("edge", "balanced", "accurate")


class ANPR:
    """Automatic number-plate recognition.

    Parameters
    ----------
    profile : "edge" | "balanced" | "accurate"
    region : str | Region
        Plate grammar for validation/correction (``"in"``, ``"uk"``, ``"eu"``,
        ``"us"``… , a JSON/YAML path, or a :class:`Region`).  ``"generic"``
        only normalises.
    detector, recognizer : optional
        Custom stage objects (see :mod:`textlens.anpr.components`), or
        ``"full-image"`` for inputs that are already plate crops.
    vehicle_detector : bool | object
        ``True`` adds an RF-DETR COCO model to report ``vehicle_type``.
    min_confidence : float
        Drop reads below this combined confidence.
    require_valid : bool
        Drop reads that do not match the region format.
    """

    def __init__(
        self,
        profile: str = "balanced",
        region: Any = "generic",
        detector: Any = None,
        recognizer: Any = None,
        vehicle_detector: Any = False,
        device: Optional[str] = None,
        min_confidence: float = 0.5,
        require_valid: bool = False,
        max_plates: int = 20,
    ) -> None:
        if profile not in ANPR_PROFILES:
            raise ConfigurationError(f"Unknown ANPR profile {profile!r}.", hint=f"Choose one of {', '.join(ANPR_PROFILES)}.")
        self.profile = profile
        self.device = device
        self.region: Region = region if isinstance(region, Region) else load_region(str(region))
        self.validator = PlateValidator(self.region)
        self.min_confidence = min_confidence
        self.require_valid = require_valid
        self.max_plates = max_plates
        self._detector = detector
        self._recognizer = recognizer
        self._vehicle = vehicle_detector
        self._tracker: Optional[PlateTracker] = None
        self.fallback_mode = False

    # ── lazily constructed stages ────────────────────────────────────────
    @property
    def detector(self) -> Any:
        if self._detector == "full-image":
            self._detector = FullImageDetector()
        if self._detector is None:
            try:
                self._detector = YoloPlateDetector(YoloPlateDetector.MODELS[self.profile], device=self.device)
            except BackendUnavailableError:
                logger.warning("open-image-models not installed; using PP-OCR text regions as plate detector (lower accuracy). %s", 'pip install "textlens-ocr[anpr]"')
                self.fallback_mode = True
                self._detector = TextRegionPlateDetector(device=self.device)
        return self._detector

    @property
    def recognizer(self) -> Any:
        if self._recognizer is None:
            try:
                self._recognizer = FastPlateRecognizer(FastPlateRecognizer.MODELS[self.profile], device=self.device)
            except BackendUnavailableError:
                logger.warning("fast-plate-ocr not installed; recognising plates with PP-OCR (lower accuracy).")
                self.fallback_mode = True
                self._recognizer = TextLensRecognizer(device=self.device)
        return self._recognizer

    @property
    def vehicle_detector(self) -> Any:
        if self._vehicle is True:
            self._vehicle = CocoVehicleDetector(device=self.device)
        return self._vehicle or None

    # ── inference ────────────────────────────────────────────────────────
    def __call__(self, image: Any, frame: Optional[int] = None) -> ANPRResult:
        from textlens.inputs.images import load_image, to_rgb
        from textlens.inputs.source import open_source

        if not hasattr(image, "size"):
            src = open_source(image)
            img = src.image if src.image is not None else load_image(src.open_binary())
        else:
            img = to_rgb(image)
        t0 = time.perf_counter()
        vehicles = self.vehicle_detector.detect(img) if self.vehicle_detector else []
        t1 = time.perf_counter()
        dets = sorted(self.detector.detect(img), key=lambda d: d.confidence, reverse=True)[: self.max_plates]
        t2 = time.perf_counter()
        crops = [crop_plate(img, d) for d in dets]
        recs = self.recognizer.recognize(crops) if crops else []
        t3 = time.perf_counter()
        plates: List[PlateRead] = []
        for det, rec in zip(dets, recs):
            v = self.validator.validate(rec.text, rec.char_confidences)
            if not (self.region.min_length <= len(v.text) <= self.region.max_length):
                continue  # empty or implausible length: not a plate reading
            combined = det.confidence * rec.confidence * (1.0 if v.valid else 0.85) * (v.score if v.valid else 1.0)
            if combined < self.min_confidence or (self.require_valid and not v.valid):
                continue
            read = PlateRead(
                plate=v.text,
                confidence=round(combined, 4),
                bbox=det.bbox.rounded(1),
                raw_text=rec.text,
                detection_confidence=round(det.confidence, 4),
                recognition_confidence=round(rec.confidence, 4),
                char_confidences=rec.char_confidences[: len(v.text)],
                polygon=det.polygon,
                valid=v.valid,
                format=v.format,
                corrections=v.corrections,
                region=rec.region,
                detector=getattr(self.detector, "name", None),
                recognizer=getattr(self.recognizer, "name", None),
            )
            self._assign_vehicle(read, vehicles)
            plates.append(read)
        timings = {"vehicles": round((t1 - t0) * 1000, 1), "detect": round((t2 - t1) * 1000, 1), "recognize": round((t3 - t2) * 1000, 1)}
        return ANPRResult(plates=plates, width=img.size[0], height=img.size[1], timings_ms=timings, profile=self.profile, frame=frame)

    @staticmethod
    def _assign_vehicle(read: PlateRead, vehicles: List[Any]) -> None:
        cx, cy = read.bbox.center
        best = None
        for v in vehicles:
            b = v.bbox
            if b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1 and (best is None or b.area < best.bbox.area):
                best = v
        if best is not None:
            read.vehicle_type = best.label
            read.vehicle_bbox = best.bbox

    def stream(self, frames: Iterable[Any], iou_threshold: float = 0.3, max_age: int = 15) -> Iterator[ANPRResult]:
        """Process video frames with tracking + per-character temporal voting."""
        tracker = PlateTracker(iou_threshold=iou_threshold, max_age=max_age)
        for i, frame in enumerate(frames):
            result = self(frame, frame=i)
            result.plates = tracker.update(i, result.plates)
            yield result

    def batch(self, images: Iterable[Any]) -> List[ANPRResult]:
        return [self(im) for im in images]

    def __repr__(self) -> str:
        return f"<ANPR profile={self.profile!r} region={self.region.name!r}>"
