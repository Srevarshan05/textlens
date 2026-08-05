"""
textlens.backends.lighton_ocr
──────────────────────────────
LightOnOCR-2-1B vision-language backend.

Follows the official LightOn HuggingFace usage pattern exactly:
  https://huggingface.co/lightonai/LightOnOCR-2-1B

HuggingFace repository : lightonai/LightOnOCR-2-1B
Cache path             : ~/.cache/textlens/models/lighton-ocr/
Requires               : transformers >= 5.0.0
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from textlens.models.base import BaseOCRModel
from textlens.models.cache import ModelCache
from textlens.models.metadata import ModelMetadata
from textlens.models.registry import ModelRegistry

logger = logging.getLogger("textlens.backends.lighton_ocr")

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
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
    _TRANSFORMERS = True
except ImportError:
    _TRANSFORMERS = False


class LightOnOCRBackend(BaseOCRModel):
    """Concrete backend for LightOnOCR-2-1B.

    Implements the full :class:`~textlens.models.base.BaseOCRModel` interface
    following the official HuggingFace usage pattern.

    Parameters
    ----------
    device : str, optional
        Target inference device (``"cuda"`` or ``"cpu"``).
        Auto-detected from CUDA availability when omitted.
    """

    def __init__(self, device: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(device=device)
        self._processor: Any = None
        self._model: Any = None

        if device is None:
            self._device = "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"

    # ------------------------------------------------------------------
    # BaseOCRModel interface
    # ------------------------------------------------------------------

    def metadata(self) -> ModelMetadata:
        return ModelRegistry.get("lighton-ocr")

    def is_installed(self) -> bool:
        return _cache.is_installed("lighton-ocr")

    def device_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 4.0,
            "cpu_supported": True,
            "recommended": "4 GB+ VRAM or CPU",
        }

    def download(self) -> None:
        from textlens.models.downloader import ModelDownloader
        dl = ModelDownloader(cache=_cache)
        dl.download("lighton-ocr")

    def load(self) -> None:
        if self._loaded:
            return

        if not _TRANSFORMERS:
            raise ImportError(
                "transformers >= 5.0.0 is required to load LightOnOCR-2.\n"
                "Run: pip install 'transformers>=5.0.0'"
            )
        if not _TORCH:
            raise ImportError("torch is required. Run: pip install torch")

        model_path = _cache.model_path("lighton-ocr")
        source = str(model_path) if model_path.exists() else "lightonai/LightOnOCR-2-1B"

        # Official dtype: bfloat16 on CUDA, float32 on CPU
        torch_dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        logger.info("Loading LightOnOCR processor from %s ...", source)
        self._processor = LightOnOcrProcessor.from_pretrained(
            source,
            fix_mistral_regex=True,  # Fix incorrect Mistral tokenizer regex pattern
        )

        logger.info("Loading LightOnOCR model on %s (dtype=%s) ...", self._device, torch_dtype)
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,  # Suppress mistral3→lighton_ocr architecture warning
        ).to(self._device)

        self._model.eval()
        self._loaded = True
        logger.info("LightOnOCR backend ready on %s", self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Any,
        prompt: str = "",
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """Run LightOnOCR inference on a single image.

        Parameters
        ----------
        image : PIL.Image | str | Path
            Image to process. Can be a PIL Image, a file path, or a URL string.
        prompt : str, optional
            Ignored — LightOnOCR-2 uses a fixed chat template for OCR.
        max_new_tokens : int, optional
            Maximum number of output tokens. Defaults to ``1024``.

        Returns
        -------
        str
            Extracted text content.
        """
        if not self._loaded:
            self.load()

        if not _PIL:
            raise ImportError("Pillow is required. Run: pip install pillow")

        start = time.time()

        # ------------------------------------------------------------------
        # Build the image source for apply_chat_template
        # Official pattern uses {"type": "image", "url": <path_or_url>}
        # We pass the PIL image directly for local files.
        # ------------------------------------------------------------------
        if isinstance(image, PILImage.Image):
            pil_img = image.convert("RGB")
            image_content: Any = {"type": "image", "image": pil_img}
        elif isinstance(image, (str, Path)):
            src = str(image)
            p = Path(src)
            if p.exists():
                pil_img = PILImage.open(p).convert("RGB")
                image_content = {"type": "image", "image": pil_img}
            else:
                # Treat as a URL
                image_content = {"type": "image", "url": src}
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # ------------------------------------------------------------------
        # Build chat conversation and tokenise
        # Official pattern from the README
        # ------------------------------------------------------------------
        conversation = [{"role": "user", "content": [image_content]}]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensors to device + dtype (official pattern)
        torch_dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        inputs = {
            k: v.to(device=self._device, dtype=torch_dtype) if v.is_floating_point() else v.to(self._device)
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[1]

        # ------------------------------------------------------------------
        # Generate — official pattern
        # ------------------------------------------------------------------
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Slice off the prompt tokens — keep only newly generated tokens
        generated_ids = output_ids[0, input_len:]
        result = self._processor.decode(generated_ids, skip_special_tokens=True)

        elapsed = round(time.time() - start, 3)
        logger.debug("LightOnOCR inference completed in %.3fs", elapsed)
        return result.strip()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"<LightOnOCRBackend device={self._device!r} {status}>"
