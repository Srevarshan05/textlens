"""
textlens.models.metadata
────────────────────────
Canonical dataclass for TextLens model metadata.

Every officially supported model in the TextLens catalog must have a
``ModelMetadata`` instance.  This is the single source of truth for model
information — it is consumed by the registry, manager, downloader, doctor
and CLI without any duplication.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class ModelMetadata:
    """Immutable descriptor for a TextLens-supported OCR / Vision model.

    Attributes
    ----------
    id : str
        Canonical lowercase slug used everywhere (e.g. ``"glm-ocr"``).
    display_name : str
        Human-readable name (e.g. ``"GLM OCR"``).
    category : str
        Broad model category (``"OCR"``, ``"Vision Language Model"``, etc.).
    parameters : str
        Parameter count as a human-readable string (``"0.9B"``, ``"770M"``).
    use_cases : list[str]
        Bullet-point use-cases shown in ``ModelManager.models()``.
    min_vram_gb : float
        Minimum VRAM (GB) required for GPU inference.
        Set to ``0.0`` for CPU-only / lightweight models.
    min_recommendation : str
        Free-text hardware recommendation shown in the CLI.
    cpu_supported : bool
        Whether the model can run on CPU (no CUDA GPU required).
    is_default : bool
        Whether this model is loaded by ``OCR()`` when no ``model=`` is given.
    hf_repo_id : str
        HuggingFace repository identifier used for downloading
        (e.g. ``"THUDM/glm-ocr"``).
    description : str
        Short one-sentence description shown by ``textlens model info``.
    download_size_gb : Optional[float]
        Approximate download size in GB (``None`` if unknown).
    """

    id: str
    display_name: str
    category: str
    parameters: str
    use_cases: List[str]
    min_vram_gb: float
    min_recommendation: str
    cpu_supported: bool
    is_default: bool
    hf_repo_id: str
    description: str
    download_size_gb: Optional[float] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def use_cases_str(self, sep: str = "\n  ") -> str:
        """Return formatted use-cases string."""
        return sep.join(self.use_cases)

    def short_label(self) -> str:
        """Return a compact label suitable for list views."""
        tag = " [Default]" if self.is_default else ""
        return f"{self.display_name}{tag} ({self.parameters})"
