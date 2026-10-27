"""
tests.test_registry
────────────────────
Unit tests for the ModelRegistry and ModelMetadata.
"""

from __future__ import annotations

import pytest

from textlens.models.registry import ModelRegistry
from textlens.models.metadata import ModelMetadata
from textlens.models.exceptions import UnknownModelError


EXPECTED_IDS = {
    "glm-ocr",
    "lighton-ocr",
    "hunyuan-ocr",
    "smolvlm",
}


class TestModelRegistry:
    def test_all_contains_legacy_models(self):
        # 2.0 grew the catalog; every 0.x model is still registered.
        models = ModelRegistry.all()
        assert len(models) >= 4

    def test_supported_ids_are_correct(self):
        ids = set(ModelRegistry.supported_ids())
        assert EXPECTED_IDS <= ids
        assert "ppocrv6-small" in ids

    def test_get_returns_metadata(self):
        meta = ModelRegistry.get("glm-ocr")
        assert isinstance(meta, ModelMetadata)
        assert meta.id == "glm-ocr"
        assert meta.display_name == "GLM-OCR"

    def test_get_normalized_aliases(self):
        assert ModelRegistry.get("LightOnOCR").id == "lighton-ocr"
        assert ModelRegistry.get("lighton-ocr").id == "lighton-ocr"
        assert ModelRegistry.get("GLM-OCR").id == "glm-ocr"
        assert ModelRegistry.get("GLM OCR").id == "glm-ocr"
        assert ModelRegistry.get("SmolVLM").id == "smolvlm"

    def test_get_unknown_raises_unknown_model_error(self):
        with pytest.raises(UnknownModelError) as exc_info:
            ModelRegistry.get("gpt-vision")
        assert "gpt-vision" in str(exc_info.value)
        assert "glm-ocr" in str(exc_info.value)

    def test_default_is_cpu_friendly_engine(self):
        # 2.0 default runs everywhere (ONNX PP-OCRv6); VLMs are routed to.
        default = ModelRegistry.default()
        assert default.id == "ppocrv6-small"
        assert default.is_default is True

    def test_only_one_default(self):
        defaults = [m for m in ModelRegistry.all() if m.is_default]
        assert len(defaults) == 1

    def test_is_registered_true(self):
        assert ModelRegistry.is_registered("glm-ocr") is True

    def test_is_registered_false(self):
        assert ModelRegistry.is_registered("some-random-model") is False

    @pytest.mark.parametrize("model_id", list(EXPECTED_IDS))
    def test_every_model_has_required_fields(self, model_id: str):
        meta = ModelRegistry.get(model_id)
        assert meta.id, "id must not be empty"
        assert meta.display_name, "display_name must not be empty"
        assert meta.category, "category must not be empty"
        assert meta.parameters, "parameters must not be empty"
        assert len(meta.use_cases) > 0, "use_cases must not be empty"
        assert meta.hf_repo_id, "hf_repo_id must not be empty"
        assert isinstance(meta.cpu_supported, bool)
        assert isinstance(meta.min_vram_gb, float)

    def test_glm_ocr_min_vram(self):
        meta = ModelRegistry.get("glm-ocr")
        assert meta.min_vram_gb == 4.0
        assert ModelRegistry.spec("glm-ocr").recommended_vram_gb == 6.0

    def test_smolvlm_min_vram(self):
        meta = ModelRegistry.get("smolvlm")
        assert meta.min_vram_gb == 2.0

    def test_unknown_model_error_message_contains_supported_ids(self):
        exc = UnknownModelError("bad-model", ["glm-ocr", "smolvlm"])
        msg = str(exc)
        assert "bad-model" in msg
        assert "glm-ocr" in msg
        assert "smolvlm" in msg
