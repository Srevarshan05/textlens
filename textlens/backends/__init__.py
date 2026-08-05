"""
textlens.backends
──────────────────
Concrete model backend implementations.

Each backend lives in its own module and implements the
``textlens.models.base.BaseOCRModel`` interface.

Currently registered backends
------------------------------
- ``glm_ocr`` → GLMOCRBackend (default)

Future backends should be added here without touching the public OCR API.
"""

from __future__ import annotations
