"""TextLens public API.

This module deliberately uses lazy exports: commands such as ``textlens
models`` only need the static catalog and must not import PyTorch,
Transformers, or FastAPI. Public imports remain backwards compatible::

    from textlens import OCR, TextLens, ModelManager
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__version__ = "0.1.0"
__author__ = "TextLens Contributors"

# (module, attribute) pairs are imported only when the public name is used.
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Modern OCR API
    "OCR": ("textlens.ocr", "OCR"),
    # Legacy GLM-OCR SDK and REST server
    "TextLens": ("textlens.sdk", "TextLens"),
    "create_app": ("textlens.server", "create_app"),
    "serve": ("textlens.server", "serve"),
    # Legacy hardware/dependency helpers
    "HardwareInfo": ("textlens.hardware", "HardwareInfo"),
    "SystemCUDADetails": ("textlens.hardware", "SystemCUDADetails"),
    "is_cuda_available": ("textlens.hardware", "is_cuda_available"),
    "detect_system_cuda": ("textlens.hardware", "detect_system_cuda"),
    "get_pytorch_cuda_install_cmd": ("textlens.hardware", "get_pytorch_cuda_install_cmd"),
    "get_hardware_info": ("textlens.hardware", "get_hardware_info"),
    "print_hardware_status": ("textlens.hardware", "print_hardware_status"),
    "DependencyReport": ("textlens.dependencies", "DependencyReport"),
    "check_dependencies": ("textlens.dependencies", "check_dependencies"),
    "ensure_dependencies": ("textlens.dependencies", "ensure_dependencies"),
    # Model catalog and management
    "ModelMetadata": ("textlens.models.metadata", "ModelMetadata"),
    "ModelRegistry": ("textlens.models.registry", "ModelRegistry"),
    "ModelManager": ("textlens.models.manager", "ModelManager"),
    "ModelCache": ("textlens.models.cache", "ModelCache"),
    "ModelDownloader": ("textlens.models.downloader", "ModelDownloader"),
    "HardwareDoctor": ("textlens.models.doctor", "HardwareDoctor"),
    "HardwareProfile": ("textlens.models.hardware", "HardwareProfile"),
    "inspect_hardware": ("textlens.models.hardware", "inspect_hardware"),
    "DiscoveredModel": ("textlens.models.discovery", "DiscoveredModel"),
    "discover_models": ("textlens.models.discovery", "discover_models"),
    "BaseOCRModel": ("textlens.models.base", "BaseOCRModel"),
    "TextLensError": ("textlens.models.exceptions", "TextLensError"),
    "UnknownModelError": ("textlens.models.exceptions", "UnknownModelError"),
    "ModelNotInstalledError": ("textlens.models.exceptions", "ModelNotInstalledError"),
    "DownloadError": ("textlens.models.exceptions", "DownloadError"),
    # Batch API
    "BatchOCR": ("textlens.batch.engine", "BatchOCR"),
    "BatchStatus": ("textlens.batch.types", "BatchStatus"),
    "BatchTask": ("textlens.batch.types", "BatchTask"),
    "TaskStatus": ("textlens.batch.types", "TaskStatus"),
    "JobMetrics": ("textlens.batch.types", "JobMetrics"),
    "BatchJobConfig": ("textlens.batch.types", "BatchJobConfig"),
    "BaseBatchQueue": ("textlens.batch.queue", "BaseBatchQueue"),
    "MemoryBatchQueue": ("textlens.batch.queue", "MemoryBatchQueue"),
    "StructuredExporter": ("textlens.batch.exporter", "StructuredExporter"),
}


def __getattr__(name: str) -> Any:
    """Resolve a public API symbol without eagerly importing optional stacks."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value  # Cache the resolved symbol for later access.
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to interactive completion tools."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["__version__", *_LAZY_EXPORTS]
