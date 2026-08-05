"""
textlens.models.base
─────────────────────
Abstract base class that every TextLens OCR / Vision backend must implement.

Design Goals
------------
- Enforce a consistent API surface across all model backends.
- Make future model additions a matter of implementing this interface and
  registering the model in the central registry — nothing else changes.
- Keep the contract minimal: ``load``, ``predict``, ``download``,
  ``is_installed``, ``metadata``, and ``device_requirements``.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Union

from textlens.models.metadata import ModelMetadata


class BaseOCRModel(abc.ABC):
    """Abstract base class for all TextLens OCR / Vision model backends.

    Subclasses must implement every abstract method.  The ``OCR`` public API
    relies exclusively on this interface, which means any backend can be
    swapped transparently.

    Parameters
    ----------
    device : str, optional
        Target inference device — ``"cuda"`` or ``"cpu"``.  When ``None``
        the backend should auto-select based on availability.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._device: Optional[str] = device
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into memory.

        This method must be idempotent — calling it multiple times must not
        reload the model if it is already resident in memory.
        """

    @abc.abstractmethod
    def predict(
        self,
        image: Any,
        prompt: str = "Text Recognition:",
        **kwargs: Any,
    ) -> str:
        """Run inference on a single image and return extracted text.

        Parameters
        ----------
        image :
            Input image.  Backends accept a ``PIL.Image.Image``, a file-path
            string, or a URL string.
        prompt : str
            Instruction prompt passed to the model.
        **kwargs :
            Backend-specific keyword arguments.

        Returns
        -------
        str
            Extracted text from the image.
        """

    @abc.abstractmethod
    def download(self) -> None:
        """Download model weights to the local TextLens cache.

        If the model is already cached, this method must be a no-op (it
        should print a confirmation message but not re-download).
        """

    @abc.abstractmethod
    def is_installed(self) -> bool:
        """Return ``True`` if the model weights are present in local cache."""

    @abc.abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return the ``ModelMetadata`` instance for this backend."""

    @abc.abstractmethod
    def device_requirements(self) -> Dict[str, Any]:
        """Return a dictionary describing the hardware requirements.

        The dictionary **must** contain at least:

        ``min_vram_gb`` : float
            Minimum GPU VRAM required in gigabytes.
        ``cpu_supported`` : bool
            Whether the model can run on CPU only.

        Additional keys are allowed for backend-specific information.
        """

    # ------------------------------------------------------------------
    # Concrete helpers (may be overridden)
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` if the model has been loaded into memory."""
        return self._loaded

    @property
    def device(self) -> Optional[str]:
        """The device this backend will use (``"cuda"`` / ``"cpu"`` / ``None``)."""
        return self._device

    def __repr__(self) -> str:
        meta = self.metadata()
        status = "loaded" if self._loaded else "unloaded"
        return f"<{self.__class__.__name__} id={meta.id!r} device={self._device!r} {status}>"
