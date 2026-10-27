"""
textlens.ocr
────────────
Backwards-compatible import location for :class:`OCR`.

The engine lives in :mod:`textlens.core.engine` since TextLens 2.0.
``from textlens.ocr import OCR`` and ``from textlens import OCR`` return the
same class.  0.x call patterns still work::

    ocr = OCR(model="glm-ocr", device="cuda")
    text = ocr.read("invoice.png")          # -> str

and 2.0 adds structured results::

    result = OCR()("invoice.pdf")           # -> textlens.Result
"""

from __future__ import annotations

from textlens.core.engine import OCR

__all__ = ["OCR"]
