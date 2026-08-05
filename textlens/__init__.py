"""
TextLens - High-Performance Python OCR Framework
=================================================

An easy-to-use Python SDK and 1-line REST endpoint server for reading text,
tables, formulas, and structured documents using GLM-OCR with GPU auto-detection.

Hardware Check & CUDA Setup:
    >>> import textlens
    >>> print(textlens.is_cuda_available())
    True
    >>> textlens.print_hardware_status()

Dependency Auto-Healing:
    >>> textlens.ensure_dependencies(auto_install=True)

Quickstart SDK:
    >>> from textlens import TextLens
    >>> ocr = TextLens()
    >>> text = ocr.read("invoice.png")

Serve REST API Endpoint:
    >>> import textlens
    >>> textlens.serve(port=8000)
"""

__version__ = "0.1.0"
__author__ = "TextLens Contributors"

from textlens.hardware import (
    HardwareInfo,
    SystemCUDADetails,
    is_cuda_available,
    detect_system_cuda,
    get_pytorch_cuda_install_cmd,
    get_hardware_info,
    print_hardware_status,
)
from textlens.dependencies import (
    DependencyReport,
    check_dependencies,
    ensure_dependencies,
)
from textlens.sdk import TextLens
from textlens.server import create_app, serve

__all__ = [
    "TextLens",
    "HardwareInfo",
    "SystemCUDADetails",
    "DependencyReport",
    "is_cuda_available",
    "detect_system_cuda",
    "get_pytorch_cuda_install_cmd",
    "get_hardware_info",
    "print_hardware_status",
    "check_dependencies",
    "ensure_dependencies",
    "create_app",
    "serve",
    "__version__",
]
