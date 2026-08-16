"""Tests for live-catalog enrichment without contacting Hugging Face."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from textlens.models.discovery import discover_models
from textlens.models.hardware import HardwareProfile


def _profile(vram: float = 8.0) -> HardwareProfile:
    return HardwareProfile(
        os_name="Test OS", python_version="3.12", torch_version="test",
        cuda_available=True, cuda_version="12.0", system_cuda_version="12.0",
        gpus=[], primary_gpu_name="Test GPU", primary_vram_gb=vram,
        cpu_name="Test CPU", cpu_physical_cores=4, cpu_logical_cores=8,
        ram_total_gb=16.0, device_type="cuda",
    )


class _FakeHfApi:
    calls = []
    models = []

    def list_models(self, **kwargs):
        self.__class__.calls.append(kwargs)
        return iter(self.__class__.models)

    def get_safetensors_metadata(self, _repo_id, timeout=6):
        return SimpleNamespace(parameter_count={"F32": 2_000_000_000})


def test_discover_models_enriches_hub_metadata():
    _FakeHfApi.calls = []
    _FakeHfApi.models = [
        SimpleNamespace(
            modelId="acme/invoice-1B-ocr",
            tags=["ocr", "document", "table", "vision"],
            pipeline_tag="image-to-text", downloads=1234, likes=5,
            safetensors={"total": 1_000_000_000},
        )
    ]
    with patch("huggingface_hub.HfApi", _FakeHfApi):
        result = discover_models(
            search="ocr", use_case="invoices", limit=5, profile=_profile()
        )

    assert _FakeHfApi.calls[0]["search"] == "ocr invoices"
    assert result[0].repo_id == "acme/invoice-1B-ocr"
    assert result[0].parameter_count_b == 1.0
    assert result[0].estimated_vram_gb == 3.0
    assert result[0].compatibility == "Compatible"
    assert "Invoices" in result[0].use_case_signals
    assert "Tables" in result[0].use_case_signals


def test_compatible_only_removes_models_that_exceed_vram():
    _FakeHfApi.calls = []
    _FakeHfApi.models = [
        SimpleNamespace(modelId="acme/large-8B-ocr", tags=["ocr"], safetensors={"total": 8_000_000_000}),
        SimpleNamespace(modelId="acme/small-1B-ocr", tags=["ocr"], safetensors={"total": 1_000_000_000}),
    ]
    with patch("huggingface_hub.HfApi", _FakeHfApi):
        result = discover_models(compatible_only=True, limit=5, profile=_profile())

    assert [item.repo_id for item in result] == ["acme/small-1B-ocr"]
    # Discovery deliberately keeps the first live request bounded. Repeated
    # searches are served from a short-lived local cache.
    assert _FakeHfApi.calls[0]["limit"] == 5


def test_discover_models_uses_exact_official_catalog_requirements():
    _FakeHfApi.models = [
        SimpleNamespace(modelId="zai-org/GLM-OCR", tags=["ocr"], safetensors=None)
    ]
    with patch("huggingface_hub.HfApi", _FakeHfApi):
        result = discover_models(limit=1, profile=_profile(vram=6.0))

    assert result[0].parameter_count_b == 0.9
    assert result[0].estimated_vram_gb == 6.0
    assert result[0].compatibility == "Compatible"
    assert "Invoices" in result[0].use_case_signals


def test_discover_models_hides_unknown_parameters_unless_requested():
    _FakeHfApi.models = [SimpleNamespace(modelId="acme/unpublished-ocr", tags=["ocr"], safetensors=None)]
    with patch("huggingface_hub.HfApi", _FakeHfApi), patch.object(
        _FakeHfApi, "get_safetensors_metadata", side_effect=RuntimeError("no metadata")
    ):
        hidden = discover_models(limit=1, profile=_profile())
        visible = discover_models(limit=1, include_unknown=True, profile=_profile())

    assert hidden == []
    assert visible[0].parameter_count_b is None
    assert visible[0].compatibility == "VRAM not published"


def test_include_unknown_keeps_unverified_models_visible_with_compatible_filter():
    _FakeHfApi.models = [SimpleNamespace(modelId="acme/unpublished-ocr", tags=["ocr"], safetensors=None)]
    with patch("huggingface_hub.HfApi", _FakeHfApi), patch.object(
        _FakeHfApi, "get_safetensors_metadata", side_effect=RuntimeError("no metadata")
    ):
        result = discover_models(
            limit=1,
            compatible_only=True,
            include_unknown=True,
            profile=_profile(),
        )

    assert result[0].compatibility == "VRAM not published"
