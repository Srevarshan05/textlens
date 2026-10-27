"""
textlens.backends
─────────────────
OCR engine adapters.  Every adapter implements
:class:`textlens.backends.base.OCRBackend` and is referenced from a model
spec (``adapter="module:Class"``); nothing here is imported until a model
is used.

Bundled adapters
    onnx.ppocr.PPOCRBackend            PP-OCRv6 on ONNX Runtime (edge / default)
    transformers_vlm.*Adapter          GLM-OCR, LightOnOCR, HunyuanOCR, SmolVLM
    openai_compat.OpenAICompatibleBackend  any OpenAI-compatible server (vLLM…)
"""

from __future__ import annotations
