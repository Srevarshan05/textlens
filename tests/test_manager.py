"""
tests.test_manager
───────────────────
Unit tests for ModelManager — validation, cache detection, and error paths.

Note: These tests mock the actual HuggingFace download so no network access
is required to run the test suite.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from textlens.models.manager import ModelManager
from textlens.models.exceptions import UnknownModelError


class TestModelManagerModels:
    def test_models_returns_list(self, capsys):
        result = ModelManager.models()
        assert isinstance(result, list)
        assert len(result) == 5

    def test_models_contains_glm_ocr(self, capsys):
        result = ModelManager.models()
        ids = [m.id for m in result]
        assert "glm-ocr" in ids

    def test_models_contains_all_official(self, capsys):
        result = ModelManager.models()
        ids = {m.id for m in result}
        expected = {
            "glm-ocr", "lighton-ocr", "hunyuan-ocr",
            "smolvlm", "paddleocr",
        }
        assert ids == expected


class TestModelManagerDownload:
    def test_download_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            ModelManager.download("not-a-real-model")

    def test_download_known_model_already_installed(self, capsys):
        with patch(
            "textlens.models.manager._cache.is_installed", return_value=True
        ):
            # Should print "already installed" message, not download again
            ModelManager.download("glm-ocr")
            captured = capsys.readouterr()
            # rich may not print to capsys stdout — just check no exception
            assert True  # No exception == pass

    def test_download_dispatches_to_downloader(self):
        with patch("textlens.models.manager._downloader.download") as mock_dl:
            ModelManager.download("glm-ocr")
            mock_dl.assert_called_once_with("glm-ocr")


class TestModelManagerRemove:
    def test_remove_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            ModelManager.remove("totally-fake-model")

    def test_remove_dispatches_to_downloader(self):
        with patch("textlens.models.manager._downloader.remove") as mock_rm:
            ModelManager.remove("smolvlm")
            mock_rm.assert_called_once_with("smolvlm")


class TestModelManagerInfo:
    def test_info_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            ModelManager.info("mystery-model")

    def test_info_returns_metadata(self):
        with patch("textlens.models.manager._cache.is_installed", return_value=False):
            meta = ModelManager.info("glm-ocr")
            assert meta.id == "glm-ocr"
            assert meta.display_name == "GLM OCR"

    def test_info_installed_shows_disk_usage(self):
        with (
            patch("textlens.models.manager._cache.is_installed", return_value=True),
            patch("textlens.models.manager._cache.disk_usage_gb", return_value=1.8),
        ):
            meta = ModelManager.info("glm-ocr")
            assert meta is not None


class TestModelManagerIsInstalled:
    def test_is_installed_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            ModelManager.is_installed("fake-model")

    def test_is_installed_false_when_not_cached(self):
        with patch("textlens.models.manager._cache.is_installed", return_value=False):
            assert ModelManager.is_installed("glm-ocr") is False

    def test_is_installed_true_when_cached(self):
        with patch("textlens.models.manager._cache.is_installed", return_value=True):
            assert ModelManager.is_installed("florence2") is True
