"""
tests.test_doctor
──────────────────
Unit tests for the HardwareDoctor deterministic recommendation rules.

All hardware profiles are constructed manually — no actual hardware detection
runs during tests.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from textlens.models.doctor import (
    HardwareDoctor,
    Recommendation,
    _evaluate_recommendations,
    _rule_for_model,
)
from textlens.models.hardware import HardwareProfile, GPUInfo


def _make_profile(
    vram_gb: float = 0.0,
    cuda: bool = False,
    gpu_name: str = "Test GPU",
    ram_gb: float = 16.0,
) -> HardwareProfile:
    """Build a fake HardwareProfile for testing."""
    gpus = []
    if vram_gb > 0 or cuda:
        gpus = [GPUInfo(index=0, name=gpu_name, vram_total_gb=vram_gb, vram_free_gb=vram_gb, cuda_available=cuda)]

    return HardwareProfile(
        os_name="Test OS",
        python_version="3.12.0",
        torch_version="2.5.0",
        cuda_available=cuda,
        cuda_version="12.4" if cuda else None,
        system_cuda_version="12.4" if cuda else None,
        gpus=gpus,
        primary_gpu_name=gpu_name if gpus else None,
        primary_vram_gb=vram_gb,
        cpu_name="Test CPU",
        cpu_physical_cores=8,
        cpu_logical_cores=16,
        ram_total_gb=ram_gb,
        device_type="cuda" if cuda else "cpu",
    )


class TestDeterministicRules:
    """Test _rule_for_model against fixed VRAM/CUDA conditions."""

    # ── PaddleOCR — always excellent ───────────────────────────────────
    def test_paddleocr_always_excellent_cpu(self):
        rec, _ = _rule_for_model("paddleocr", vram_gb=0.0, cuda_available=False)
        assert rec == Recommendation.EXCELLENT

    def test_paddleocr_always_excellent_gpu(self):
        rec, _ = _rule_for_model("paddleocr", vram_gb=8.0, cuda_available=True)
        assert rec == Recommendation.EXCELLENT

    # ── SmolVLM ────────────────────────────────────────────────────────
    def test_smolvlm_excellent_on_cpu(self):
        rec, _ = _rule_for_model("smolvlm", vram_gb=0.0, cuda_available=False)
        assert rec == Recommendation.EXCELLENT

    def test_smolvlm_excellent_with_2gb_vram(self):
        rec, _ = _rule_for_model("smolvlm", vram_gb=2.0, cuda_available=True)
        assert rec == Recommendation.EXCELLENT

    def test_smolvlm_supported_with_low_vram_cuda(self):
        rec, _ = _rule_for_model("smolvlm", vram_gb=1.0, cuda_available=True)
        assert rec == Recommendation.SUPPORTED

    # ── Florence-2 ─────────────────────────────────────────────────────
    def test_florence2_excellent_with_4gb(self):
        rec, _ = _rule_for_model("florence2", vram_gb=4.0, cuda_available=True)
        assert rec == Recommendation.EXCELLENT

    def test_florence2_supported_on_cpu(self):
        rec, _ = _rule_for_model("florence2", vram_gb=0.0, cuda_available=False)
        assert rec == Recommendation.SUPPORTED

    def test_florence2_supported_with_2gb(self):
        rec, _ = _rule_for_model("florence2", vram_gb=2.0, cuda_available=True)
        assert rec == Recommendation.SUPPORTED

    def test_florence2_not_recommended_with_1gb(self):
        rec, _ = _rule_for_model("florence2", vram_gb=1.0, cuda_available=True)
        assert rec == Recommendation.NOT_RECOMMENDED

    # ── GLM OCR ────────────────────────────────────────────────────────
    def test_glm_ocr_excellent_with_6gb(self):
        rec, _ = _rule_for_model("glm-ocr", vram_gb=6.0, cuda_available=True)
        assert rec == Recommendation.EXCELLENT

    def test_glm_ocr_excellent_with_8gb(self):
        rec, _ = _rule_for_model("glm-ocr", vram_gb=8.0, cuda_available=True)
        assert rec == Recommendation.EXCELLENT

    def test_glm_ocr_supported_with_4gb(self):
        rec, _ = _rule_for_model("glm-ocr", vram_gb=4.0, cuda_available=True)
        assert rec == Recommendation.SUPPORTED

    def test_glm_ocr_supported_on_cpu(self):
        rec, _ = _rule_for_model("glm-ocr", vram_gb=0.0, cuda_available=False)
        assert rec == Recommendation.SUPPORTED

    def test_glm_ocr_not_recommended_low_vram(self):
        rec, _ = _rule_for_model("glm-ocr", vram_gb=1.0, cuda_available=True)
        assert rec == Recommendation.NOT_RECOMMENDED

    # ── LightOnOCR ─────────────────────────────────────────────────────
    def test_lighton_excellent_with_8gb(self):
        rec, _ = _rule_for_model("lighton-ocr", vram_gb=8.0, cuda_available=True)
        assert rec == Recommendation.EXCELLENT

    def test_lighton_supported_with_6gb(self):
        rec, _ = _rule_for_model("lighton-ocr", vram_gb=6.0, cuda_available=True)
        assert rec == Recommendation.SUPPORTED

    def test_lighton_not_recommended_with_4gb(self):
        rec, _ = _rule_for_model("lighton-ocr", vram_gb=4.0, cuda_available=True)
        assert rec == Recommendation.NOT_RECOMMENDED

    # ── HunyuanOCR ─────────────────────────────────────────────────────
    def test_hunyuan_excellent_with_8gb(self):
        rec, _ = _rule_for_model("hunyuan-ocr", vram_gb=8.0, cuda_available=True)
        assert rec == Recommendation.EXCELLENT

    def test_hunyuan_not_recommended_on_cpu(self):
        rec, _ = _rule_for_model("hunyuan-ocr", vram_gb=0.0, cuda_available=False)
        assert rec == Recommendation.NOT_RECOMMENDED

    def test_hunyuan_not_recommended_with_4gb(self):
        rec, _ = _rule_for_model("hunyuan-ocr", vram_gb=4.0, cuda_available=True)
        assert rec == Recommendation.NOT_RECOMMENDED


class TestEvaluateRecommendations:
    def test_returns_one_recommendation_per_model(self):
        profile = _make_profile(vram_gb=8.0, cuda=True)
        recs = _evaluate_recommendations(profile)
        assert len(recs) == 6

    def test_all_excellent_with_8gb(self):
        profile = _make_profile(vram_gb=8.0, cuda=True)
        recs = _evaluate_recommendations(profile)
        levels = {r.model.id: r.level for r in recs}
        # All 6 models should be at least Supported with 8GB
        for rec in recs:
            assert rec.level != Recommendation.NOT_RECOMMENDED, (
                f"{rec.model.id} should not be NOT_RECOMMENDED with 8GB VRAM"
            )

    def test_cpu_only_paddleocr_smolvlm_excellent(self):
        profile = _make_profile(vram_gb=0.0, cuda=False)
        recs = _evaluate_recommendations(profile)
        levels = {r.model.id: r.level for r in recs}
        assert levels["paddleocr"] == Recommendation.EXCELLENT
        assert levels["smolvlm"] == Recommendation.EXCELLENT
        # Hunyuan not recommended on CPU
        assert levels["hunyuan-ocr"] == Recommendation.NOT_RECOMMENDED

    def test_4gb_glm_ocr_supported(self):
        profile = _make_profile(vram_gb=4.0, cuda=True)
        recs = _evaluate_recommendations(profile)
        levels = {r.model.id: r.level for r in recs}
        assert levels["glm-ocr"] == Recommendation.SUPPORTED
        assert levels["florence2"] == Recommendation.EXCELLENT
        assert levels["smolvlm"] == Recommendation.EXCELLENT


class TestHardwareDoctor:
    def test_run_returns_doctor_report(self):
        from textlens.models.doctor import DoctorReport
        profile = _make_profile(vram_gb=6.0, cuda=True)
        with patch("textlens.models.doctor.inspect_hardware", return_value=profile):
            doctor = HardwareDoctor()
            report = doctor.run()
            assert isinstance(report, DoctorReport)
            assert report.profile == profile
            assert len(report.recommendations) == 6

    def test_print_report_does_not_raise(self, capsys):
        from textlens.models.doctor import DoctorReport
        profile = _make_profile(vram_gb=4.0, cuda=True)
        recs = _evaluate_recommendations(profile)
        report = DoctorReport(profile=profile, recommendations=recs)
        doctor = HardwareDoctor()
        # Should not raise regardless of rich availability
        doctor.print_report(report)
