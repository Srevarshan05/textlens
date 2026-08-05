"""
textlens.models
───────────────
Public API surface for the TextLens model management subsystem.

Exports
-------
ModelMetadata
    Immutable descriptor for a registered TextLens model.

ModelRegistry
    Central read-only registry of all officially supported models.

ModelManager
    Developer-facing class for listing, downloading, and removing models.

ModelCache
    Local disk cache manager (advanced use).

ModelDownloader
    HuggingFace downloader (advanced use).

HardwareDoctor
    Deterministic hardware inspector and model recommender.

HardwareProfile
    Immutable snapshot of system hardware capabilities.

Exceptions
----------
TextLensError, UnknownModelError, ModelNotInstalledError,
DownloadError, HardwareInspectionError
"""

from __future__ import annotations

from textlens.models.metadata import ModelMetadata
from textlens.models.registry import ModelRegistry
from textlens.models.cache import ModelCache
from textlens.models.downloader import ModelDownloader
from textlens.models.manager import ModelManager
from textlens.models.hardware import HardwareProfile, inspect_hardware
from textlens.models.doctor import HardwareDoctor, DoctorReport, ModelRecommendation, Recommendation
from textlens.models.base import BaseOCRModel
from textlens.models.exceptions import (
    TextLensError,
    UnknownModelError,
    ModelNotInstalledError,
    DownloadError,
    HardwareInspectionError,
)

__all__ = [
    # Metadata & Registry
    "ModelMetadata",
    "ModelRegistry",
    # Management
    "ModelCache",
    "ModelDownloader",
    "ModelManager",
    # Hardware & Doctor
    "HardwareProfile",
    "inspect_hardware",
    "HardwareDoctor",
    "DoctorReport",
    "ModelRecommendation",
    "Recommendation",
    # Base interface
    "BaseOCRModel",
    # Exceptions
    "TextLensError",
    "UnknownModelError",
    "ModelNotInstalledError",
    "DownloadError",
    "HardwareInspectionError",
]
