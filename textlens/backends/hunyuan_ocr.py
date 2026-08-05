"""
textlens.backends.hunyuan_ocr
──────────────────────────────
HunyuanOCR-1.5 vision-language backend by Tencent.

Follows the official Tencent/HunyuanOCR HuggingFace usage pattern:
  https://huggingface.co/tencent/HunyuanOCR

HuggingFace repository : tencent/HunyuanOCR
Cache path             : ~/.cache/textlens/models/hunyuan-ocr/
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

logger = logging.getLogger("textlens.backends.hunyuan_ocr")

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
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
    _TRANSFORMERS = True
except ImportError:
    _TRANSFORMERS = False

DEFAULT_HUNYUAN_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


class HunyuanOCRBackend(BaseOCRModel):
    """Concrete backend for Tencent HunyuanOCR.

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
        return ModelRegistry.get("hunyuan-ocr")

    def is_installed(self) -> bool:
        return _cache.is_installed("hunyuan-ocr")

    def device_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 8.0,
            "cpu_supported": True,
            "recommended": "8 GB VRAM or CPU",
        }

    def download(self) -> None:
        from textlens.models.downloader import ModelDownloader
        dl = ModelDownloader(cache=_cache)
        dl.download("hunyuan-ocr")

    def load(self) -> None:
        if self._loaded:
            return

        if not _TRANSFORMERS:
            raise ImportError(
                "transformers >= 5.0.0 is required to load HunyuanOCR.\n"
                "Run: pip install 'transformers>=5.0.0'"
            )
        if not _TORCH:
            raise ImportError("torch is required. Run: pip install torch")

        model_path = _cache.model_path("hunyuan-ocr")
        source = str(model_path) if model_path.exists() else "tencent/HunyuanOCR"

        torch_dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        logger.info("Loading HunyuanOCR processor from %s ...", source)
        self._processor = AutoProcessor.from_pretrained(
            source,
            trust_remote_code=True,
            use_fast=False,
        )

        logger.info("Loading HunyuanOCR model on %s (dtype=%s) ...", self._device, torch_dtype)
        self._model = HunYuanVLForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self._device)

        self._model.eval()
        self._loaded = True
        logger.info("HunyuanOCR backend ready on %s", self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Any,
        prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Run HunyuanOCR inference on a single image.

        Parameters
        ----------
        image : PIL.Image | str | Path
            Image to process.
        prompt : str, optional
            Custom instruction prompt. If omitted or empty, uses default
            HunyuanOCR document parsing prompt.
        max_new_tokens : int, optional
            Maximum output tokens. Defaults to ``4096``.

        Returns
        -------
        str
            Extracted text/markdown content.
        """
        if not self._loaded:
            self.load()

        if not _PIL:
            raise ImportError("Pillow is required. Run: pip install pillow")

        start = time.time()

        if isinstance(image, PILImage.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            p = Path(str(image))
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {p}")
            pil_img = PILImage.open(p).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        text_prompt = prompt if (prompt and prompt.strip()) else DEFAULT_HUNYUAN_PROMPT

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": text_prompt},
            ],
        }]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gen_ids = output_ids[:, input_len:]
        result = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        # Post-process: strip bounding box coordinates e.g. (4,22),(996,121)
        import re
        clean_text = re.sub(r'(\(\d{1,4},\s*\d{1,4}\),?\s*)+', ' ', result)
        clean_text = re.sub(r'[ \t]{2,}', ' ', clean_text)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

        elapsed = round(time.time() - start, 3)
        logger.debug("HunyuanOCR inference completed in %.3fs", elapsed)
        return clean_text.strip()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"<HunyuanOCRBackend device={self._device!r} {status}>"
