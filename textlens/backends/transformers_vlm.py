"""
textlens.backends.transformers_vlm
──────────────────────────────────
Adapters exposing the local document VLMs (PyTorch / Transformers) through
the TextLens backend contract.

The model-specific inference code is unchanged from TextLens 0.x and lives
in the per-model modules (``glm_ocr.py``, ``lighton_ocr.py`` …).  These
adapters only add what 2.0 needs around it: task → prompt mapping from the
model spec, deterministic decoding, and conversion of the generated
Markdown into :class:`~textlens.backends.base.PageOCR` (the pipeline then
parses it into blocks, tables and formulas).

``torch`` / ``transformers`` are imported only when a model actually loads.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from textlens.backends.base import OCRBackend, PageOCR, RecognizeOptions


class _LegacyVLMAdapter(OCRBackend):
    """Wrap a 0.x ``BaseOCRModel`` backend (``predict(image, prompt) -> str``)."""

    backend_name = "transformers"
    legacy_path: str = ""  # "module:Class"
    default_prompt: Optional[str] = None
    generation_defaults: Dict[str, Any] = {}

    def _load(self) -> None:
        import importlib

        module_name, _, cls_name = self.legacy_path.partition(":")
        cls = getattr(importlib.import_module(module_name), cls_name)
        device = self.requested_device if self.requested_device not in (None, "auto") else None
        self._impl = cls(device=device, **{k: v for k, v in self.options.items() if k == "torch_dtype"})
        self._impl.load()
        self.device = getattr(self._impl, "device", None) or getattr(self._impl, "_device", None)

    def _unload(self) -> None:
        impl = getattr(self, "_impl", None)
        self._impl = None
        if impl is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _prompt(self, options: RecognizeOptions) -> Optional[str]:
        if options.prompt:
            return options.prompt
        prompts = self.spec.prompts or {}
        return prompts.get(options.task) or prompts.get("text") or self.default_prompt

    def _recognize(self, images: List[Any], options: RecognizeOptions) -> List[PageOCR]:
        results: List[PageOCR] = []
        kwargs = dict(self.generation_defaults)
        if options.max_new_tokens:
            kwargs["max_new_tokens"] = options.max_new_tokens
        prompt = self._prompt(options)
        for img in images:
            t0 = time.perf_counter()
            call_kwargs = dict(kwargs)
            if prompt is not None:
                call_kwargs["prompt"] = prompt
            text = self._impl.predict(img, **call_kwargs)
            results.append(
                PageOCR(
                    width=img.size[0],
                    height=img.size[1],
                    markdown=(text or "").strip(),
                    confidence=None,  # generative output: no calibrated score
                    timings_ms={"generate": round((time.perf_counter() - t0) * 1000, 1)},
                )
            )
        return results


class GLMOCRAdapter(_LegacyVLMAdapter):
    legacy_path = "textlens.backends.glm_ocr:GLMOCRBackend"
    default_prompt = "Text Recognition:"
    generation_defaults = {"max_new_tokens": 4096, "temperature": 0.0}


class LightOnOCRAdapter(_LegacyVLMAdapter):
    legacy_path = "textlens.backends.lighton_ocr:LightOnOCRBackend"
    default_prompt = ""
    generation_defaults = {"max_new_tokens": 4096}


class HunyuanOCRAdapter(_LegacyVLMAdapter):
    legacy_path = "textlens.backends.hunyuan_ocr:HunyuanOCRBackend"
    default_prompt = None  # the backend's own document-parsing prompt
    generation_defaults = {"max_new_tokens": 4096}


class SmolVLMAdapter(_LegacyVLMAdapter):
    legacy_path = "textlens.backends.smolvlm:SmolVLMBackend"
    default_prompt = "Extract all text from this image:"
    generation_defaults = {"max_new_tokens": 1024}
