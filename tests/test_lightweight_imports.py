"""Regression checks for fast CLI/catalog imports.

The package must not import the VLM or REST stacks merely to print the model
catalog. Those dependencies are intentionally optional and expensive.
"""

from __future__ import annotations

import subprocess
import sys


def test_top_level_import_defers_heavy_dependencies():
    code = (
        "import sys, textlens; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'fastapi' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_model_manager_import_defers_huggingface_client():
    code = (
        "import sys; "
        "from textlens.models.manager import ModelManager; "
        "assert ModelManager; "
        "assert 'huggingface_hub' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
