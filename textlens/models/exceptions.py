"""
textlens.models.exceptions
──────────────────────────
Custom exception types for the TextLens model subsystem.

All exceptions inherit from ``TextLensError`` so callers can catch broadly
or specifically as needed.
"""

from __future__ import annotations

from typing import List


class TextLensError(Exception):
    """Base exception for all TextLens errors."""


class UnknownModelError(TextLensError):
    """Raised when an unregistered model ID is requested.

    Parameters
    ----------
    model_id : str
        The unrecognised model identifier the caller requested.
    supported_ids : list[str]
        The full list of officially registered model IDs.

    Example
    -------
    ::

        Unknown model: "gpt-vision"

        Supported models are:
          • glm-ocr        (Default)
          • lighton-ocr
          • hunyuan-ocr
          • florence2
          • smolvlm
          • paddleocr
    """

    def __init__(self, model_id: str, supported_ids: List[str]) -> None:
        self.model_id = model_id
        self.supported_ids = supported_ids
        supported_str = "\n  • ".join(supported_ids)
        super().__init__(
            f'\nUnknown model: "{model_id}"\n\n'
            f"Supported models are:\n  • {supported_str}\n"
        )


class ModelNotInstalledError(TextLensError):
    """Raised when an installed model is expected but not found on disk.

    This is an internal guard — the public ``OCR`` class auto-downloads
    missing models before raising this error.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            f'Model "{model_id}" is not installed. '
            f"Run: textlens model install {model_id}"
        )


class DownloadError(TextLensError):
    """Raised when a model download fails."""

    def __init__(self, model_id: str, reason: str) -> None:
        self.model_id = model_id
        super().__init__(f'Failed to download "{model_id}": {reason}')


class HardwareInspectionError(TextLensError):
    """Raised when hardware inspection cannot complete."""
