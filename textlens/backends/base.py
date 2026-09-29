"""
textlens.backends.base
──────────────────────
The adapter contract every OCR engine implements.

An adapter turns images into :class:`PageOCR` — engine output in the pixel
space of each input image.  Detection-style engines (PP-OCR) return
positioned ``items`` (text lines with boxes and confidences); generative
engines (document VLMs) return ``markdown``.  The pipeline converts either
form into the unified :class:`~textlens.core.result.Page`, so nothing
model-specific leaks past this boundary.

Writing a new adapter::

    class MyBackend(OCRBackend):
        thread_safe = True                     # may run concurrently

        def _load(self):                       # heavy imports go here
            import mylib
            self.model = mylib.load(self.model_path)

        def _recognize(self, images, options):
            return [PageOCR(items=[...], width=im.width, height=im.height) for im in images]

Register it by giving a :class:`~textlens.models.specs.ModelSpec` an
``adapter="package.module:MyBackend"`` string.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from textlens.documents.layout import Item
from textlens.models.specs import ModelSpec

logger = logging.getLogger("textlens.backends")


@dataclass
class RecognizeOptions:
    """Per-call options understood by adapters (unknown keys are ignored)."""

    task: str = "text"  # text | table | formula | markdown | extraction | chart
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None
    language: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageOCR:
    """Raw engine output for one image (pixel coordinates)."""

    width: int
    height: int
    items: List[Item] = field(default_factory=list)  # positioned text lines
    markdown: Optional[str] = None  # generative engines
    confidence: Optional[float] = None
    model: Optional[str] = None
    model_revision: Optional[str] = None
    backend: Optional[str] = None
    device: Optional[str] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    raw: Any = None

    @property
    def text(self) -> str:
        if self.markdown is not None:
            return self.markdown
        return "\n".join(i.text for i in self.items)


class OCRBackend(abc.ABC):
    """Base class for OCR engine adapters."""

    #: Set to True when ``recognize`` may be called from several threads at
    #: once (ONNX Runtime sessions, HTTP clients).  GPU PyTorch models are
    #: serialised with a per-instance lock.
    thread_safe: bool = False
    backend_name: str = "unknown"

    def __init__(self, spec: ModelSpec, device: Optional[str] = None, **options: Any) -> None:
        self.spec = spec
        self.requested_device = device
        self.device: Optional[str] = device
        self.options = options
        self._loaded = False
        self._load_lock = threading.Lock()
        self._infer_lock = threading.RLock()
        self.load_time_ms: Optional[float] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self.spec.id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load weights (idempotent, thread-safe)."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            self._load()
            self.load_time_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            self._loaded = True
            logger.info("Loaded %s on %s in %.0f ms", self.spec.id, self.device, self.load_time_ms)

    def unload(self) -> None:
        with self._load_lock:
            if self._loaded:
                self._unload()
                self._loaded = False

    def _unload(self) -> None:
        """Release resources; default drops references for the GC."""

    def recognize(self, images: Sequence[Any], options: Optional[RecognizeOptions] = None) -> List[PageOCR]:
        """Recognise text in PIL images; returns one :class:`PageOCR` each."""
        self.load()
        opts = options or RecognizeOptions()
        if self.thread_safe:
            results = self._recognize(list(images), opts)
        else:
            with self._infer_lock:
                results = self._recognize(list(images), opts)
        for r in results:
            r.model = r.model or self.spec.id
            r.model_revision = r.model_revision or self.spec.revision
            r.backend = r.backend or self.backend_name
            r.device = r.device or self.device
        return results

    @abc.abstractmethod
    def _load(self) -> None: ...

    @abc.abstractmethod
    def _recognize(self, images: List[Any], options: RecognizeOptions) -> List[PageOCR]: ...

    # ── conveniences ─────────────────────────────────────────────────────
    def warmup(self) -> None:
        from PIL import Image

        self.recognize([Image.new("RGB", (64, 32), "white")])

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.spec.id,
            "backend": self.backend_name,
            "device": self.device,
            "loaded": self._loaded,
            "load_time_ms": self.load_time_ms,
            "thread_safe": self.thread_safe,
        }

    def __enter__(self) -> "OCRBackend":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} model={self.spec.id!r} device={self.device!r} loaded={self._loaded}>"
