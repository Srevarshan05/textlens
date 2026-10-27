"""
textlens.models.exceptions
──────────────────────────
Backwards-compatible home of the model-subsystem exceptions.

The canonical definitions now live in :mod:`textlens.errors`; they are
re-exported here so ``from textlens.models.exceptions import …`` keeps
working and the classes stay identical (``except`` clauses match either way).
"""

from __future__ import annotations

from textlens.errors import (  # noqa: F401  (re-exports)
    DownloadError,
    HardwareInspectionError,
    IntegrityError,
    ModelError,
    ModelNotInstalledError,
    OfflineError,
    TextLensError,
    UnknownModelError,
)

__all__ = [
    "TextLensError",
    "ModelError",
    "UnknownModelError",
    "ModelNotInstalledError",
    "DownloadError",
    "IntegrityError",
    "OfflineError",
    "HardwareInspectionError",
]
