"""
textlens.backends.glm_ocr
──────────────────────────
GLM-OCR backend — the TextLens default OCR engine (0.9B).

This backend wraps the original TextLens GLM-OCR inference logic from
``textlens.sdk.TextLens`` inside the ``BaseOCRModel`` interface so it can
be managed uniformly through the model registry, downloader, and OCR API.

HuggingFace repository : THUDM/glm-ocr
Cache path             : ~/.cache/textlens/models/glm-ocr/
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from textlens.models.base import BaseOCRModel
from textlens.models.cache import ModelCache
from textlens.models.metadata import ModelMetadata
from textlens.models.registry import ModelRegistry

logger = logging.getLogger("textlens.backends.glm_ocr")

_cache = ModelCache()

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from PIL import Image as PILImage

    _PIL = True
except ImportError:
    _PIL = False

try:
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    _TRANSFORMERS = True
except ImportError:
    _TRANSFORMERS = False

try:
    from textlens.progress import ProgressTracker
    _PROGRESS = True
except ImportError:
    _PROGRESS = False


class GLMOCRBackend(BaseOCRModel):
    """Concrete backend for the GLM-OCR 0.9B model.

    Implements the full :class:`~textlens.models.base.BaseOCRModel` interface.

    Parameters
    ----------
    device : str, optional
        ``"cuda"`` or ``"cpu"``.  Auto-selected when ``None``.
    torch_dtype : torch.dtype, optional
        Precision for loading weights.  Defaults to ``bfloat16`` / ``float16``
        on CUDA and ``float32`` on CPU.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device)
        self._torch_dtype = torch_dtype
        self._processor: Any = None
        self._model: Any = None

        # Resolve device
        if device is None:
            if _TORCH and torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

    # ------------------------------------------------------------------
    # BaseOCRModel interface
    # ------------------------------------------------------------------

    def metadata(self) -> ModelMetadata:
        """Return the registry metadata for this backend."""
        return ModelRegistry.get("glm-ocr")

    def is_installed(self) -> bool:
        """Return ``True`` if the model weights are cached on disk."""
        return _cache.is_installed("glm-ocr")

    def device_requirements(self) -> Dict[str, Any]:
        """Return hardware requirements for GLM-OCR."""
        return {
            "min_vram_gb": 6.0,
            "cpu_supported": True,
            "recommended": "6 GB VRAM GPU",
        }

    def download(self) -> None:
        """Download GLM-OCR weights to cache."""
        from textlens.models.downloader import ModelDownloader
        dl = ModelDownloader(cache=_cache)
        dl.download("glm-ocr")

    def load(self) -> None:
        """Load processor and model weights into memory."""
        if self._loaded:
            return

        if not _TRANSFORMERS:
            raise ImportError(
                "transformers is required to load GLM-OCR.\n"
                "Run: pip install transformers accelerate"
            )
        if not _TORCH:
            raise ImportError("torch is required. Run: pip install torch")

        model_path = _cache.model_path("glm-ocr")

        # Resolve torch dtype
        if self._torch_dtype is None:
            if self._device == "cuda" and torch.cuda.is_available():
                self._torch_dtype = (
                    torch.bfloat16
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                self._torch_dtype = torch.float32

        if _PROGRESS:
            progress = ProgressTracker(100, desc=f"Loading GLM-OCR ({self._device.upper()})")
            progress.update(20, "Initializing Processor...")
        else:
            progress = None
            print(f"Loading GLM-OCR on {self._device.upper()} ...")

        # Load from local cache if available, otherwise from HuggingFace hub
        source = str(model_path) if model_path.exists() else "zai-org/GLM-OCR"

        self._processor = AutoProcessor.from_pretrained(source)

        if progress:
            progress.update(50, "Loading Model Weights...")

        if self._device == "cuda" and torch.cuda.is_available():
            self._model = GlmOcrForConditionalGeneration.from_pretrained(
                source,
                device_map="auto",
                torch_dtype=self._torch_dtype,
            )
        else:
            self._model = GlmOcrForConditionalGeneration.from_pretrained(
                source,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
            )

        if progress:
            progress.update(85, "Setting Model to Eval Mode...")

        self._model.eval()
        self._loaded = True

        active_dev = next(self._model.parameters()).device
        if progress:
            progress.complete(f"Loaded on {active_dev}")
        else:
            print(f"✓ GLM-OCR loaded on {active_dev}")

        logger.info("GLM-OCR backend loaded on %s", active_dev)

    def predict(
        self,
        image: Any,
        prompt: str = "Text Recognition:",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        dpi: int = 200,
        page: Optional[Union[int, List[int]]] = None,
        **kwargs: Any,
    ) -> str:
        """Run GLM-OCR inference on an image or PDF document.

        Parameters
        ----------
        image : Any
            PIL Image, image file path, PDF file path, URL string, or bytes.
        prompt : str, optional
            Instruction prompt for the model.
        max_new_tokens : int, optional
            Maximum number of tokens to generate. Defaults to ``512``.
        temperature : float, optional
            Sampling temperature (0 = greedy). Defaults to ``0.7``.
        top_p : float, optional
            Nucleus sampling probability. Defaults to ``0.95``.
        dpi : int, optional
            DPI resolution when rendering PDF pages. Defaults to ``200``.
        page : int | list[int], optional
            Page number(s) to process if input is a PDF. Defaults to all pages.

        Returns
        -------
        str
            Extracted text from the image or PDF.
        """
        if not self._loaded:
            self.load()

        from textlens.utils.image_utils import load_input_images

        images = load_input_images(image, dpi=dpi, page=page)
        start = time.time()

        page_results = []
        for pil_img in images:
            content_item: Dict[str, Any] = {"type": "image", "image": pil_img}
            messages = [
                {
                    "role": "user",
                    "content": [content_item, {"type": "text", "text": prompt}],
                }
            ]

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    do_sample=temperature > 0,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[:, input_len:]
            result = self._processor.decode(generated[0], skip_special_tokens=True).strip()
            page_results.append(result)

        elapsed = round(time.time() - start, 3)
        logger.debug("GLM-OCR inference completed in %.3fs for %d page(s)", elapsed, len(images))

        if len(page_results) == 1:
            return page_results[0]

        formatted_pages = [f"--- Page {idx} ---\n{text}" for idx, text in enumerate(page_results, start=1)]
        return "\n\n".join(formatted_pages)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"<GLMOCRBackend device={self._device!r} {status}>"
