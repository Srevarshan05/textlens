"""
TextLens — Multi-Model OCR Framework
======================================

A modular, production-ready Python OCR framework supporting multiple
officially registered models through a unified API.

Quickstart
----------
::

    from textlens import OCR

    # Default model (GLM-OCR)
    ocr = OCR()
    text = ocr.read("invoice.png")

    # Switch model
    ocr = OCR(model="smolvlm")

Model Management
----------------
::

    from textlens import ModelManager

    ModelManager.models()
    ModelManager.download("glm-ocr")
    ModelManager.info("florence2")

Hardware Doctor
---------------
::

    from textlens.models import HardwareDoctor

    doctor = HardwareDoctor()
    report = doctor.run()
    doctor.print_report(report)

Legacy SDK (GLM-OCR only)
-------------------------
::

    from textlens import TextLens
    ocr = TextLens()
    text = ocr.read("sample.png")
"""

from __future__ import annotations

__version__ = "0.2.3"
__author__ = "TextLens Contributors"

# ── Legacy hardware API (backwards compatible) ──────────────────────────────
from textlens.hardware import (
    HardwareInfo,
    SystemCUDADetails,
    is_cuda_available,
    detect_system_cuda,
    get_pytorch_cuda_install_cmd,
    get_hardware_info,
    print_hardware_status,
)

# ── Legacy dependency API (backwards compatible) ────────────────────────────
from textlens.dependencies import (
    DependencyReport,
    check_dependencies,
    ensure_dependencies,
)

# ── Legacy SDK (backwards compatible) ───────────────────────────────────────
from textlens.sdk import TextLens

# ── Server (backwards compatible) ───────────────────────────────────────────
from textlens.server import create_app, serve

# ── New Model Registry API ──────────────────────────────────────────────────
from textlens.models import (
    ModelMetadata,
    ModelRegistry,
    ModelManager,
    ModelCache,
    ModelDownloader,
    HardwareDoctor,
    HardwareProfile,
    inspect_hardware,
    BaseOCRModel,
    # Exceptions
    TextLensError,
    UnknownModelError,
    ModelNotInstalledError,
    DownloadError,
)

# ── High-level OCR API ──────────────────────────────────────────────────────
from textlens.ocr import OCR

# ── Batch Processing API ────────────────────────────────────────────────────
from textlens.batch import (
    BatchOCR,
    BatchStatus,
    BatchTask,
    TaskStatus,
    JobMetrics,
    BatchJobConfig,
    BaseBatchQueue,
    MemoryBatchQueue,
    StructuredExporter,
)


__all__ = [
    # ── Batch API ──────────────────────────────────────────────────────
    "BatchOCR",
    "BatchStatus",
    "BatchTask",
    "TaskStatus",
    "JobMetrics",
    "BatchJobConfig",
    "BaseBatchQueue",
    "MemoryBatchQueue",
    "StructuredExporter",
    # ── New API ────────────────────────────────────────────────────────
    "OCR",
    "ModelMetadata",
    "ModelRegistry",
    "ModelManager",
    "ModelCache",
    "ModelDownloader",
    "HardwareDoctor",
    "HardwareProfile",
    "inspect_hardware",
    "BaseOCRModel",
    # Exceptions
    "TextLensError",
    "UnknownModelError",
    "ModelNotInstalledError",
    "DownloadError",
    # ── Legacy API ─────────────────────────────────────────────────────
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
