"""
TextLens - Python OCR Framework wrapping zai-org/GLM-OCR
=========================================================

An easy-to-use Python SDK and REST server framework for reading text,
tables, formulas, and structured document JSON. Works on GPU or CPU.

Quickstart:
    >>> from textlens import TextLens
    >>> ocr = TextLens()
    >>> text = ocr.read("invoice.png")

Serve REST API:
    >>> import textlens
    >>> textlens.serve(port=8000)
"""

__version__ = "0.1.0"
__author__ = "Z.ai & TextLens Contributors"

from textlens.hardware import (
    HardwareInfo,
    get_hardware_info,
    print_hardware_status,
)
from textlens.sdk import TextLens
from textlens.server import create_app, serve

__all__ = [
    "TextLens",
    "HardwareInfo",
    "get_hardware_info",
    "print_hardware_status",
    "create_app",
    "serve",
    "__version__",
]
