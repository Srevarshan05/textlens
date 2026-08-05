"""
tests.test_cli
───────────────
Integration tests for the TextLens CLI argument parser and command routing.

These tests only verify that the parser constructs correctly and routes to
the right handlers — they do not trigger downloads or hardware detection.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from argparse import Namespace

import sys


def _run_cli(*args: str) -> None:
    """Run the CLI with given args, temporarily overriding sys.argv."""
    from textlens.cli import main
    with patch.object(sys, "argv", ["textlens", *args]):
        main()


class TestCLIModels:
    def test_models_command_calls_manager(self):
        with patch("textlens.cli._cmd_models") as mock:
            with patch.object(sys, "argv", ["textlens", "models"]):
                from textlens.cli import main
                main()
            mock.assert_called_once()


class TestCLIModelInstall:
    def test_model_install_calls_download(self):
        with patch("textlens.models.manager.ModelManager.download") as mock_dl:
            with patch.object(sys, "argv", ["textlens", "model", "install", "glm-ocr"]):
                from textlens.cli import main
                main()
            mock_dl.assert_called_once_with("glm-ocr")

    def test_model_install_unknown_exits_with_error(self):
        from textlens.models.exceptions import UnknownModelError
        with patch(
            "textlens.models.manager.ModelManager.download",
            side_effect=UnknownModelError("bad-model", ["glm-ocr"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                with patch.object(sys, "argv", ["textlens", "model", "install", "bad-model"]):
                    from textlens.cli import main
                    main()
            assert exc_info.value.code == 1


class TestCLIModelRemove:
    def test_model_remove_calls_remove(self):
        with patch("textlens.models.manager.ModelManager.remove") as mock_rm:
            with patch.object(sys, "argv", ["textlens", "model", "remove", "smolvlm"]):
                from textlens.cli import main
                main()
            mock_rm.assert_called_once_with("smolvlm")

    def test_model_remove_unknown_exits_with_error(self):
        from textlens.models.exceptions import UnknownModelError
        with patch(
            "textlens.models.manager.ModelManager.remove",
            side_effect=UnknownModelError("bad-model", ["glm-ocr"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                with patch.object(sys, "argv", ["textlens", "model", "remove", "bad-model"]):
                    from textlens.cli import main
                    main()
            assert exc_info.value.code == 1


class TestCLIModelInfo:
    def test_model_info_calls_info(self):
        from textlens.models.registry import ModelRegistry
        mock_meta = ModelRegistry.get("glm-ocr")
        with patch(
            "textlens.models.manager.ModelManager.info",
            return_value=mock_meta,
        ):
            with patch.object(sys, "argv", ["textlens", "model", "info", "glm-ocr"]):
                from textlens.cli import main
                main()


class TestCLIDoctor:
    def test_doctor_command_calls_doctor(self):
        with patch("textlens.cli._cmd_doctor") as mock_doctor:
            with patch.object(sys, "argv", ["textlens", "doctor"]):
                from textlens.cli import main
                main()
            mock_doctor.assert_called_once()


class TestCLIVersion:
    def test_version_flag_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["textlens", "--version"]):
                from textlens.cli import main
                main()
        assert exc_info.value.code == 0
