"""TextLens — OCR without the OCR complexity.

    >>> from textlens import OCR
    >>> print(OCR()("document.pdf").text)

The public API is exported lazily: ``import textlens`` does not import
PyTorch, Transformers, FastAPI or ONNX Runtime.  Each name below is
resolved on first use, so the CLI and model catalog stay fast.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from textlens._version import __version__

__author__ = "TextLens Contributors"

# (module, attribute) pairs are imported only when the public name is used.
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # ── 2.0 core API ───────────────────────────────────────────────────
    "OCR": ("textlens.core.engine", "OCR"),
    "Result": ("textlens.core.result", "Result"),
    "Page": ("textlens.core.result", "Page"),
    "Block": ("textlens.core.result", "Block"),
    "Line": ("textlens.core.result", "Line"),
    "Word": ("textlens.core.result", "Word"),
    "Table": ("textlens.core.result", "Table"),
    "Formula": ("textlens.core.result", "Formula"),
    "BBox": ("textlens.core.result", "BBox"),
    "Chunk": ("textlens.core.result", "Chunk"),
    "Document": ("textlens.documents.document", "Document"),
    "load": ("textlens.documents.document", "load"),
    "inspect_pdf": ("textlens.documents.pdf.inspector", "inspect_pdf"),
    "Router": ("textlens.core.router", "Router"),
    "PROFILES": ("textlens.core.router", "PROFILES"),
    "ModelSpec": ("textlens.models.specs", "ModelSpec"),
    "register_spec": ("textlens.models.specs", "register_spec"),
    "OCRBackend": ("textlens.backends.base", "OCRBackend"),
    "inspect_system": ("textlens.runtime.system", "inspect_system"),
    "ANPR": ("textlens.anpr.pipeline", "ANPR"),
    "get_settings": ("textlens.config", "get_settings"),
    # Errors
    "TextLensError": ("textlens.errors", "TextLensError"),
    "UnknownModelError": ("textlens.errors", "UnknownModelError"),
    "ModelNotInstalledError": ("textlens.errors", "ModelNotInstalledError"),
    "DownloadError": ("textlens.errors", "DownloadError"),
    "BackendUnavailableError": ("textlens.errors", "BackendUnavailableError"),
    "RoutingError": ("textlens.errors", "RoutingError"),
    # ── 0.x APIs kept for backward compatibility ───────────────────────
    "TextLens": ("textlens.sdk", "TextLens"),
    "create_app": ("textlens.serving.app", "create_app"),
    "serve": ("textlens.serving.app", "serve"),
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

# ``textlens.models`` is also a subpackage; expose the model API namespace.
_SUBMODULES = {"models": "textlens.models"}


def __getattr__(name: str) -> Any:
    """Resolve a public API symbol without eagerly importing optional stacks."""
    if name in _SUBMODULES:
        value = import_module(_SUBMODULES[name])
    else:
        try:
            module_name, attribute = _LAZY_EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        value = getattr(import_module(module_name), attribute)
    globals()[name] = value  # cache for later access
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to interactive completion tools."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_SUBMODULES))


__all__ = ["__version__", *_LAZY_EXPORTS]
