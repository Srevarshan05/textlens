"""
textlens.backends.smolvlm
─────────────────────────
SmolVLM-256M vision-language OCR backend.

HuggingFace repository : HuggingFaceTB/SmolVLM-256M-Instruct
Cache path             : ~/.cache/textlens/models/smolvlm/
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

logger = logging.getLogger("textlens.backends.smolvlm")

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
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLMModel
    except ImportError:
        try:
            from transformers import AutoModelForConditionalGeneration as AutoVLMModel
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoVLMModel
    _TRANSFORMERS = True
except ImportError:
    _TRANSFORMERS = False


class SmolVLMBackend(BaseOCRModel):
    """Concrete backend for the SmolVLM-256M model.

    Implements the full :class:`~textlens.models.base.BaseOCRModel` interface.

    Parameters
    ----------
    device : str, optional
        Target inference device (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, device: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(device=device)
        self._processor: Any = None
        self._model: Any = None

        if device is None:
            if _TORCH and torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

    def metadata(self) -> ModelMetadata:
        return ModelRegistry.get("smolvlm")

    def is_installed(self) -> bool:
        return _cache.is_installed("smolvlm")

    def device_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 2.0,
            "cpu_supported": True,
            "recommended": "2 GB VRAM or CPU",
        }

    def download(self) -> None:
        from textlens.models.downloader import ModelDownloader

        dl = ModelDownloader(cache=_cache)
        dl.download("smolvlm")

    def load(self) -> None:
        if self._loaded:
            return

        if not _TRANSFORMERS:
            raise ImportError(
                "transformers is required to load SmolVLM.\n"
                "Run: pip install transformers accelerate"
            )
        if not _TORCH:
            raise ImportError("torch is required. Run: pip install torch")

        model_path = _cache.model_path("smolvlm")
        source = str(model_path) if model_path.exists() else "HuggingFaceTB/SmolVLM-256M-Instruct"

        logger.info("Loading SmolVLM processor and model from %s ...", source)
        self._processor = AutoProcessor.from_pretrained(source)

        dtype = torch.bfloat16 if (self._device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float32

        if self._device == "cuda" and torch.cuda.is_available():
            self._model = AutoVLMModel.from_pretrained(
                source,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            self._model = AutoVLMModel.from_pretrained(
                source,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
            )

        self._model.eval()
        self._loaded = True
        logger.info("SmolVLM backend loaded successfully on %s", self._device)

    def predict(
        self,
        image: Any,
        prompt: str = "Extract all text from this image:",
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        if not self._loaded:
            self.load()

        if not _PIL:
            raise ImportError("Pillow is required. Run: pip install pillow")

        start = time.time()

        if isinstance(image, PILImage.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            src = str(image)
            p = Path(src)
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {p}")
            pil_img = PILImage.open(p).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text_prompt, images=[pil_img], return_tensors="pt")
        inputs = inputs.to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode output text omitting the input prompt
        input_len = inputs["input_ids"].shape[1]
        generated_texts = self._processor.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )

        elapsed = round(time.time() - start, 3)
        logger.debug("SmolVLM inference completed in %.3fs", elapsed)
        return generated_texts[0].strip()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"<SmolVLMBackend device={self._device!r} {status}>"
