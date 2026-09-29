"""
textlens.documents.pdf.pdfium
─────────────────────────────
Thin, thread-safe access layer over pypdfium2.

PDFium is **not thread-safe**: concurrent calls from several threads corrupt
its global state.  Every PDFium call in TextLens therefore runs under
:data:`PDFIUM_LOCK`.  Parallel PDF work should use processes, not threads.

Coordinates
    PDF user space has a bottom-left origin and ignores ``/Rotate``.  TextLens
    reports boxes in *display points*: top-left origin, rotation applied,
    CropBox offset removed — i.e. the same frame as a rendered page image
    divided by ``dpi / 72``.  :class:`PageTransform` converts between them
    using ``FPDF_PageToDevice`` so rotation and crop are always exact.
"""

from __future__ import annotations

import ctypes
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union

from textlens.errors import DocumentError, EncryptedDocumentError

PDFIUM_LOCK = threading.RLock()


def _pdfium():
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    return pdfium, raw


@contextmanager
def open_pdf(source: Union[str, Path, bytes], password: Optional[str] = None) -> Iterator[Any]:
    """Open a PDF under the global lock and always close it."""
    pdfium, _ = _pdfium()
    with PDFIUM_LOCK:
        try:
            doc = pdfium.PdfDocument(source if isinstance(source, bytes) else str(source), password=password)
        except pdfium.PdfiumError as exc:
            msg = str(exc)
            if "password" in msg.lower():
                raise EncryptedDocumentError(
                    "The PDF is password protected.",
                    hint="Pass password='…' (Python) or --password (CLI).",
                ) from exc
            raise DocumentError(f"Cannot open PDF: {msg}") from exc
        try:
            yield doc
        finally:
            doc.close()


@dataclass(frozen=True)
class PageTransform:
    """Affine map from PDF user space to display points (top-left origin).

    ``width``/``height`` are the *display* size (``/Rotate`` applied).
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    width: float
    height: float

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)

    def rect(self, left: float, bottom: float, right: float, top: float) -> Tuple[float, float, float, float]:
        """Map a PDF-space rectangle to a top-left ``(x0, y0, x1, y1)`` box."""
        x0, y0 = self.apply(left, bottom)
        x1, y1 = self.apply(right, top)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def apply_arrays(self, xs: Any, ys: Any) -> Tuple[Any, Any]:
        return (self.a * xs + self.c * ys + self.e, self.b * xs + self.d * ys + self.f)


def page_transform(page: Any) -> PageTransform:
    """Derive the page→display transform from three ``FPDF_PageToDevice`` probes.

    Device coordinates are integers in PDFium; probing with a large device
    size (and scaling back) keeps the transform accurate to ~0.01 pt.
    """
    _, raw = _pdfium()
    width, height = page.get_size()
    k = 100.0
    size_x, size_y = int(round(width * k)), int(round(height * k))
    out_x, out_y = ctypes.c_int(), ctypes.c_int()

    def probe(px: float, py: float) -> Tuple[float, float]:
        raw.FPDF_PageToDevice(page.raw, 0, 0, size_x, size_y, 0, px, py, ctypes.byref(out_x), ctypes.byref(out_y))
        return out_x.value / k, out_y.value / k

    ox, oy = probe(0.0, 0.0)
    xx, xy = probe(1000.0, 0.0)
    yx, yy = probe(0.0, 1000.0)
    return PageTransform(
        a=(xx - ox) / 1000.0,
        b=(xy - oy) / 1000.0,
        c=(yx - ox) / 1000.0,
        d=(yy - oy) / 1000.0,
        e=ox,
        f=oy,
        width=float(width),
        height=float(height),
    )


def render_page(page: Any, dpi: float, max_side: int = 6000, grayscale: bool = False) -> Tuple[Any, float]:
    """Render a page to an RGB PIL image; return ``(image, effective_dpi)``.

    The effective DPI is reduced when the page would exceed ``max_side``
    pixels on its longest side (posters, engineering drawings).
    """
    width, height = page.get_size()
    scale = dpi / 72.0
    longest = max(width, height) * scale
    if longest > max_side:
        scale = max_side / max(width, height)
    bitmap = page.render(scale=scale, grayscale=grayscale, may_draw_forms=True, draw_annots=True)
    img = bitmap.to_pil()
    img = img.convert("RGB") if img.mode != "RGB" else img
    return img, scale * 72.0
