"""
textlens.backends.florence2
───────────────────────────
Florence-2 Base vision foundation backend.

Follows the official Microsoft/Florence-2 HuggingFace usage pattern exactly:
  https://huggingface.co/microsoft/Florence-2-base

HuggingFace repository : microsoft/Florence-2-base
Cache path             : ~/.cache/textlens/models/florence2/

Compatibility patches applied at load-time for transformers >= 4.49
─────────────────────────────────────────────────────────────────────
1. PretrainedConfig.forced_bos_token_id missing  → patched to None
2. PreTrainedTokenizerBase.additional_special_tokens missing → patched
3. PreTrainedModel._supports_sdpa missing → patched to True (eager attn)
4. prepare_inputs_for_generation EncoderDecoderCache subscript error
   → monkey-patched to handle both legacy tuple-of-tuples and new
     EncoderDecoderCache / DynamicCache objects
"""

from __future__ import annotations

import logging
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional

from textlens.models.base import BaseOCRModel
from textlens.models.cache import ModelCache
from textlens.models.metadata import ModelMetadata
from textlens.models.registry import ModelRegistry

logger = logging.getLogger("textlens.backends.florence2")

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
    from transformers import AutoProcessor, AutoModelForCausalLM

    _TRANSFORMERS = True
except ImportError:
    _TRANSFORMERS = False


# ---------------------------------------------------------------------------
# Compatibility patch: prepare_inputs_for_generation
# ---------------------------------------------------------------------------
# Florence-2's cached remote code does:
#   past_length = past_key_values[0][0].shape[2]
# This breaks in transformers >= 4.49 where past_key_values is an
# EncoderDecoderCache / DynamicCache object, not a tuple-of-tuples.
# We replace the method on the loaded model instance with a version that
# handles both formats correctly, allowing num_beams=3 with use_cache=True
# exactly as documented by Microsoft.
# ---------------------------------------------------------------------------
def _patched_prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    past_key_values=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs,
):
    """Drop-in for Florence2ForConditionalGeneration.prepare_inputs_for_generation.

    Handles both the legacy tuple-of-tuples format and the modern
    EncoderDecoderCache / DynamicCache objects introduced in transformers 4.49+.
    """
    if past_key_values is not None:
        try:
            # Legacy format: ((key, value), ...) per layer
            past_length = past_key_values[0][0].shape[2]
        except (TypeError, KeyError, IndexError):
            # Modern format: EncoderDecoderCache wraps self + cross attention caches
            try:
                past_length = past_key_values.self_attention_cache.get_seq_length()
            except AttributeError:
                # DynamicCache (single-model greedy path)
                past_length = past_key_values.get_seq_length()

        if decoder_input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default: keep only the last token id
            remove_prefix_length = decoder_input_ids.shape[1] - 1

        decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

    return {
        "input_ids": None,  # encoder_outputs is provided; input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "use_cache": use_cache,
    }


class Florence2Backend(BaseOCRModel):
    """Concrete backend for Microsoft Florence-2 Base model.

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
        return ModelRegistry.get("florence2")

    def is_installed(self) -> bool:
        return _cache.is_installed("florence2")

    def device_requirements(self) -> Dict[str, Any]:
        return {
            "min_vram_gb": 4.0,
            "cpu_supported": True,
            "recommended": "4 GB VRAM or CPU",
        }

    def download(self) -> None:
        from textlens.models.downloader import ModelDownloader

        dl = ModelDownloader(cache=_cache)
        dl.download("florence2")

    def load(self) -> None:
        if self._loaded:
            return

        if not _TRANSFORMERS:
            raise ImportError(
                "transformers is required to load Florence-2.\n"
                "Run: pip install transformers accelerate"
            )
        if not _TORCH:
            raise ImportError("torch is required. Run: pip install torch")

        import transformers
        import transformers.modeling_utils

        # Patch 1: forced_bos_token_id (missing in some config classes)
        if not hasattr(transformers.PretrainedConfig, "forced_bos_token_id"):
            transformers.PretrainedConfig.forced_bos_token_id = None

        # Patch 2: additional_special_tokens (missing in some tokenizer classes)
        if not hasattr(transformers.PreTrainedTokenizerBase, "additional_special_tokens"):
            transformers.PreTrainedTokenizerBase.additional_special_tokens = property(
                lambda self: getattr(self, "all_special_tokens", [])
            )

        # Patch 3: _supports_sdpa (required by dispatch logic in newer transformers)
        transformers.modeling_utils.PreTrainedModel._supports_sdpa = property(
            lambda self: True
        )

        model_path = _cache.model_path("florence2")
        source = str(model_path) if model_path.exists() else "microsoft/Florence-2-base"

        logger.info("Loading Florence-2 processor from %s ...", source)
        self._processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)

        # Official pattern: float16 on CUDA, float32 on CPU
        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info("Loading Florence-2 model on %s (dtype=%s) ...", self._device, torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # Bypass SDPA vision tower dispatch error
        ).to(self._device)

        self._model.eval()

        # Patch 4: prepare_inputs_for_generation — handle EncoderDecoderCache
        # Florence-2's outer generate() delegates to self.language_model.generate().
        # The broken past_key_values subscript is in language_model's own
        # prepare_inputs_for_generation, so we must patch that inner object.
        self._model.language_model.prepare_inputs_for_generation = types.MethodType(
            _patched_prepare_inputs_for_generation, self._model.language_model
        )
        logger.debug("Applied prepare_inputs_for_generation patch for transformers >= 4.49")

        self._loaded = True
        logger.info("Florence-2 backend ready on %s", self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Any,
        prompt: str = "<OCR>",
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        **kwargs: Any,
    ) -> str:
        """Run Florence-2 OCR inference on a single image.

        Parameters
        ----------
        image : PIL.Image | str | Path
            Image to process.
        prompt : str, optional
            Florence-2 task token. Defaults to ``"<OCR>"``.
            Any free-text prompt is silently coerced to ``"<OCR>"``.
        max_new_tokens : int, optional
            Maximum number of output tokens. Defaults to ``1024``.
        num_beams : int, optional
            Beam search width. ``3`` is the official recommendation from
            Microsoft's documentation. Use ``1`` for greedy decoding.

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

        # Load and convert image
        if isinstance(image, PILImage.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            p = Path(str(image))
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {p}")
            pil_img = PILImage.open(p).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Normalize to a valid Florence-2 task token
        task_prompt = prompt.strip()
        if not (task_prompt.startswith("<") and task_prompt.endswith(">")):
            task_prompt = "<OCR>"

        # Square-pad: Florence-2's vision tower requires square feature maps.
        # Non-square images trigger an AssertionError inside the vision encoder.
        # Pad to a white square canvas while preserving the original aspect ratio.
        w, h = pil_img.size
        if w != h:
            side = max(w, h)
            canvas = PILImage.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(pil_img, ((side - w) // 2, (side - h) // 2))
            pil_img = canvas

        # Pre-process — official pattern: processor(...).to(device, dtype)
        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
        inputs = self._processor(
            text=task_prompt,
            images=pil_img,
            return_tensors="pt",
        ).to(self._device, torch_dtype)

        # Generate — official pattern (num_beams=3, no use_cache override)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Post-process structured output (Florence-2 returns tagged text)
        try:
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(pil_img.width, pil_img.height),
            )
            result = str(parsed.get(task_prompt, generated_text))
        except Exception:
            result = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        elapsed = round(time.time() - start, 3)
        logger.debug("Florence-2 inference completed in %.3fs", elapsed)
        return result.strip()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"<Florence2Backend device={self._device!r} {status}>"

