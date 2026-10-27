"""
textlens.models.registry
─────────────────────────
Backwards-compatible registry facade.

Since TextLens 2.0 the catalog lives in :mod:`textlens.models.specs`
(:class:`~textlens.models.specs.ModelSpec` carries capabilities, hardware
requirements, license and pinned artifacts).  This module keeps the 0.x
``ModelRegistry`` API, returning :class:`ModelMetadata` views derived from
the specs, so existing code keeps working unchanged.

Usage
-----
    from textlens.models.registry import ModelRegistry

    meta = ModelRegistry.get("glm-ocr")
    all_models = ModelRegistry.all()
    ids = ModelRegistry.supported_ids()
    default = ModelRegistry.default()
    spec = ModelRegistry.spec("ppocrv6-small")      # full 2.0 spec
"""

from __future__ import annotations

from typing import List

from textlens.models.metadata import ModelMetadata
from textlens.models.specs import ModelSpec, all_specs, default_spec, find_spec, get_spec


class ModelRegistry:
    """Read-only registry of all TextLens models (class-methods only)."""

    @classmethod
    def all(cls) -> List[ModelMetadata]:
        """Every registered model, in catalog order."""
        return [s.to_metadata() for s in all_specs()]

    @classmethod
    def specs(cls) -> List[ModelSpec]:
        """Every registered model as a full :class:`ModelSpec`."""
        return all_specs()

    @classmethod
    def supported_ids(cls) -> List[str]:
        return [s.id for s in all_specs()]

    @classmethod
    def get(cls, model_id: str) -> ModelMetadata:
        """Metadata by id, display name, alias or Hugging Face repo id.

        Raises :class:`~textlens.errors.UnknownModelError` if not found.
        """
        return get_spec(model_id).to_metadata()

    @classmethod
    def spec(cls, model_id: str) -> ModelSpec:
        return get_spec(model_id)

    @classmethod
    def is_registered(cls, model_id: str) -> bool:
        return find_spec(model_id) is not None

    @classmethod
    def default(cls) -> ModelMetadata:
        """The default engine (``ppocrv6-small`` since 2.0; runs on any CPU)."""
        return default_spec().to_metadata()
