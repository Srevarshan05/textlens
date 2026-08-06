"""
textlens.ocr
─────────────
High-level ``OCR`` class — the primary entry point for developers.

Design goals
------------
- Developer experience in **one line** — ``from textlens import OCR``.
- Zero manual model management: if the requested model is not cached, it is
  automatically downloaded, cached, and loaded.
- Transparent model switching: ``OCR(model="smolvlm")`` just works.
- Sensible defaults: ``OCR()`` uses the registered default model (glm-ocr).

Usage
-----
    from textlens import OCR

    # Default model (glm-ocr)
    ocr = OCR()
    text = ocr.read("invoice.png")

    # Switch model
    ocr = OCR(model="smolvlm")
    text = ocr.read("photo.jpg")

    # Force a specific device
    ocr = OCR(model="smolvlm", device="cpu")

    # Pre-download without running inference
    OCR.ensure("glm-ocr")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from textlens.models.cache import ModelCache
from textlens.models.downloader import ModelDownloader
from textlens.models.exceptions import UnknownModelError
from textlens.models.registry import ModelRegistry

logger = logging.getLogger("textlens.ocr")

_cache = ModelCache()
_downloader = ModelDownloader(cache=_cache)


class OCR:
    """Simplified OCR interface for TextLens.

    Automatically selects, downloads, and loads the requested model.

    Parameters
    ----------
    model : str, optional
        Canonical TextLens model ID (e.g. ``"glm-ocr"``, ``"smolvlm"``).
        Defaults to the registered default model (currently ``"glm-ocr"``).
    device : str, optional
        Inference device override — ``"cuda"`` or ``"cpu"``.
        When ``None`` the backend auto-selects based on availability.
    auto_download : bool
        If ``True`` (default), missing models are automatically downloaded
        before loading.  Set to ``False`` to raise immediately instead.
    **kwargs
        Additional keyword arguments forwarded to the backend constructor.

    Raises
    ------
    textlens.models.exceptions.UnknownModelError
        If *model* is not in the official TextLens catalog.
    textlens.models.exceptions.ModelNotInstalledError
        If *auto_download* is ``False`` and the model is not cached locally.

    Examples
    --------
    ::

        >>> from textlens import OCR
        >>> ocr = OCR()                        # uses glm-ocr
        >>> ocr = OCR(model="smolvlm")         # switch model
        >>> ocr = OCR(model="smolvlm", device="cpu")
        >>> text = ocr.read("document.png")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        auto_download: bool = True,
        **kwargs: Any,
    ) -> None:
        # Resolve model ID
        if model is None:
            meta = ModelRegistry.default()
            self._model_id = meta.id
        else:
            # Validate — raises UnknownModelError for unsupported IDs
            meta = ModelRegistry.get(model)
            self._model_id = meta.id

        self._meta = meta
        self._device = device
        self._auto_download = auto_download
        self._kwargs = kwargs
        self._backend: Any = None

        logger.info("OCR initialised with model=%s device=%s", self._model_id, device)

        # Ensure model is on disk
        self._ensure_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Download the model if not already cached."""
        if not _cache.is_installed(self._model_id):
            if not self._auto_download:
                from textlens.models.exceptions import ModelNotInstalledError
                raise ModelNotInstalledError(self._model_id)

            try:
                from rich.console import Console
                console = Console(force_terminal=True, highlight=False)
                console.print(
                    f"[bold yellow][Downloading] {self._meta.display_name}[/bold yellow] "
                    f"[dim]is not installed - downloading automatically...[/dim]"
                )
            except ImportError:
                print(
                    f"[Downloading] {self._meta.display_name} is not installed "
                    f"- downloading automatically..."
                )

            _downloader.download(self._model_id)

    def _load_backend(self) -> None:
        """Lazy-load the model backend on first use."""
        if self._backend is not None:
            return

        # Backend dispatch — resolved from the registry.
        # When backends are implemented they register themselves here.
        # Currently we resolve to a placeholder that calls the real SDK.
        self._backend = _resolve_backend(
            self._model_id,
            self._device,
            **self._kwargs,
        )
        self._backend.load()
        logger.info("Backend loaded: %s on %s", self._model_id, self._device)

    # ------------------------------------------------------------------
    # Public inference interface
    # ------------------------------------------------------------------

    def read(self, source: Any, prompt: str = "Text Recognition:", **kwargs: Any) -> str:
        """Extract text from an image or PDF document.

        Parameters
        ----------
        source : str | Path | PIL.Image.Image | bytes | BytesIO
            Accepted formats:

            - **Image file** — local path to ``.png``, ``.jpg``, ``.webp``, etc.
            - **PDF file** — local path to a ``.pdf`` file (multi-page supported).
            - **URL** — ``http://`` / ``https://`` pointing to an image or PDF.
            - **PIL.Image** — in-memory Pillow image object.
            - **bytes / BytesIO** — raw byte buffer of an image or PDF.
        prompt : str, optional
            Instruction prompt forwarded to the model. Defaults to
            ``"Text Recognition:"``.
        **kwargs :
            Additional keyword arguments forwarded to the backend's
            ``predict()`` method.  Common options:

            - ``dpi`` *(int)* — DPI for PDF rendering (default ``200``).
            - ``page`` *(int | list[int])* — which PDF page(s) to process
              (1-indexed, ``None`` = all pages).
            - ``max_new_tokens`` *(int)* — maximum tokens to generate.

        Returns
        -------
        str
            Extracted text.  For multi-page PDFs, pages are separated by
            ``--- Page N ---`` headers.
        """
        self._load_backend()
        return self._backend.predict(source, prompt=prompt, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """The canonical ID of the active model."""
        return self._model_id

    @property
    def model_name(self) -> str:
        """The display name of the active model."""
        return self._meta.display_name

    @property
    def device(self) -> Optional[str]:
        """The inference device (``"cuda"`` / ``"cpu"`` / ``None``)."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Whether the backend model is loaded in memory."""
        return self._backend is not None and self._backend.is_loaded

    # ------------------------------------------------------------------
    # Class-level utilities
    # ------------------------------------------------------------------

    @classmethod
    def ensure(cls, model_id: str) -> None:
        """Ensure a model is downloaded without loading it into memory.

        Useful for pre-warming cache in a setup script or container build.

        Parameters
        ----------
        model_id : str
            Canonical model slug.

        Raises
        ------
        textlens.models.exceptions.UnknownModelError
            If *model_id* is not in the official catalog.
        """
        ModelRegistry.get(model_id)  # validate
        _downloader.download(model_id)

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "idle"
        return (
            f"<OCR model={self._model_id!r} device={self._device!r} {status}>"
        )


# ---------------------------------------------------------------------------
# Backend resolver
# ---------------------------------------------------------------------------


def _resolve_backend(model_id: str, device: Optional[str], **kwargs: Any) -> Any:
    """Resolve the concrete backend class for a given model ID.

    This is the dispatch table for backend implementations.  When a new
    backend is added, register it here.  The public ``OCR`` API never
    changes.

    Parameters
    ----------
    model_id : str
        Canonical TextLens model ID.
    device : str, optional
        Target inference device.
    **kwargs :
        Forwarded to the backend constructor.

    Returns
    -------
    BaseOCRModel
        A concrete backend implementing the ``BaseOCRModel`` interface.

    Raises
    ------
    textlens.models.exceptions.UnknownModelError
        If no backend is registered for *model_id* (should not happen when
        models are properly registered).
    """
    # Lazy imports to keep startup time fast
    if model_id == "glm-ocr":
        from textlens.backends.glm_ocr import GLMOCRBackend
        return GLMOCRBackend(device=device, **kwargs)

    if model_id == "smolvlm":
        from textlens.backends.smolvlm import SmolVLMBackend
        return SmolVLMBackend(device=device, **kwargs)

    if model_id == "lighton-ocr":
        from textlens.backends.lighton_ocr import LightOnOCRBackend
        return LightOnOCRBackend(device=device, **kwargs)

    if model_id == "hunyuan-ocr":
        from textlens.backends.hunyuan_ocr import HunyuanOCRBackend
        return HunyuanOCRBackend(device=device, **kwargs)

    # For models without a dedicated backend yet, raise a helpful message
    from textlens.models.exceptions import TextLensError
    raise TextLensError(
        f'Backend for model "{model_id}" is not yet implemented.\n'
        f"The model is registered in the catalog and can be downloaded,\n"
        f"but inference support is coming in a future release.\n"
        f"For inference today, use: glm-ocr"
    )
