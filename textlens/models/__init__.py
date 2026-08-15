"""Lazy public API for TextLens model management.

Keeping this package lightweight is important because ``textlens models``
only needs catalog metadata and should not import download or GPU tooling.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ModelMetadata": ("textlens.models.metadata", "ModelMetadata"),
    "ModelRegistry": ("textlens.models.registry", "ModelRegistry"),
    "ModelCache": ("textlens.models.cache", "ModelCache"),
    "ModelDownloader": ("textlens.models.downloader", "ModelDownloader"),
    "ModelManager": ("textlens.models.manager", "ModelManager"),
    "HardwareProfile": ("textlens.models.hardware", "HardwareProfile"),
    "inspect_hardware": ("textlens.models.hardware", "inspect_hardware"),
    "DiscoveredModel": ("textlens.models.discovery", "DiscoveredModel"),
    "discover_models": ("textlens.models.discovery", "discover_models"),
    "HardwareDoctor": ("textlens.models.doctor", "HardwareDoctor"),
    "DoctorReport": ("textlens.models.doctor", "DoctorReport"),
    "ModelRecommendation": ("textlens.models.doctor", "ModelRecommendation"),
    "Recommendation": ("textlens.models.doctor", "Recommendation"),
    "BaseOCRModel": ("textlens.models.base", "BaseOCRModel"),
    "TextLensError": ("textlens.models.exceptions", "TextLensError"),
    "UnknownModelError": ("textlens.models.exceptions", "UnknownModelError"),
    "ModelNotInstalledError": ("textlens.models.exceptions", "ModelNotInstalledError"),
    "DownloadError": ("textlens.models.exceptions", "DownloadError"),
    "HardwareInspectionError": ("textlens.models.exceptions", "HardwareInspectionError"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = list(_LAZY_EXPORTS)
