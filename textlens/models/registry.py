"""
textlens.models.registry
─────────────────────────
Central Model Registry — the single authoritative source of all TextLens
officially supported models.

RULES
-----
- Only models registered here are supported.
- No model metadata is hardcoded anywhere else in the codebase.
- Future models are added here ONLY — no changes to manager, CLI, or
  downloader are required for metadata additions.

Usage
-----
    from textlens.models.registry import ModelRegistry

    meta = ModelRegistry.get("glm-ocr")
    all_models = ModelRegistry.all()
    ids = ModelRegistry.supported_ids()
    default = ModelRegistry.default()
"""

from __future__ import annotations

from typing import Dict, List, Optional

from textlens.models.metadata import ModelMetadata


# ---------------------------------------------------------------------------
# Official TextLens Model Catalog
# ---------------------------------------------------------------------------
# Each entry is immutable (frozen dataclass) and serves as the ground-truth
# for every subsystem that needs model information.

_CATALOG: List[ModelMetadata] = [
    # ── 1. GLM OCR ─────────────────────────────────────────────────────────
    ModelMetadata(
        id="glm-ocr",
        display_name="GLM OCR",
        category="OCR",
        parameters="0.9B",
        use_cases=[
            "General OCR",
            "Invoices",
            "Books",
            "Research Papers",
            "Forms",
            "Tables",
            "Formula Recognition",
            "Markdown Export",
        ],
        min_vram_gb=6.0,
        min_recommendation="6 GB VRAM",
        cpu_supported=True,
        is_default=True,
        hf_repo_id="zai-org/GLM-OCR",
        description=(
            "TextLens default OCR engine. Delivers high-accuracy text "
            "extraction across a wide range of document types including "
            "invoices, research papers, tables, and formulas."
        ),
        download_size_gb=1.8,
    ),

    # ── 2. LightOnOCR ──────────────────────────────────────────────────────
    ModelMetadata(
        id="lighton-ocr",
        display_name="LightOnOCR",
        category="OCR",
        parameters="1B",
        use_cases=[
            "Scientific Documents",
            "Academic Papers",
            "Multilingual OCR",
            "PDF Parsing",
        ],
        min_vram_gb=8.0,
        min_recommendation="8 GB VRAM",
        cpu_supported=True,
        is_default=False,
        hf_repo_id="lightonai/LightOnOCR-2-1B",
        description=(
            "Specialised OCR model optimised for scientific and academic "
            "documents with strong multilingual coverage and high-quality "
            "PDF parsing."
        ),
        download_size_gb=2.0,
    ),

    # ── 3. HunyuanOCR ──────────────────────────────────────────────────────
    ModelMetadata(
        id="hunyuan-ocr",
        display_name="HunyuanOCR",
        category="OCR",
        parameters="~1B",
        use_cases=[
            "Enterprise Documents",
            "Charts",
            "Tables",
            "Complex Layouts",
            "Information Extraction",
        ],
        min_vram_gb=8.0,
        min_recommendation="8 GB VRAM",
        cpu_supported=True,
        is_default=False,
        hf_repo_id="tencent/HunyuanOCR",
        description=(
            "Enterprise-grade OCR model by Tencent HunyuanDiT. "
            "Excels at complex document layouts, charts, and structured "
            "information extraction."
        ),
        download_size_gb=2.0,
    ),

    # ── 4. Florence-2 ──────────────────────────────────────────────────────
    ModelMetadata(
        id="florence2",
        display_name="Florence-2 Base",
        category="Vision Foundation Model",
        parameters="770M",
        use_cases=[
            "Fast OCR",
            "Captioning",
            "Detection",
            "Grounding",
            "General Vision",
        ],
        min_vram_gb=4.0,
        min_recommendation="4 GB VRAM",
        cpu_supported=True,
        is_default=False,
        hf_repo_id="microsoft/Florence-2-base",
        description=(
            "Microsoft Florence-2 vision foundation model. Supports OCR, "
            "image captioning, object detection, and visual grounding at "
            "excellent speed with low memory requirements."
        ),
        download_size_gb=1.5,
    ),

    # ── 5. SmolVLM ─────────────────────────────────────────────────────────
    ModelMetadata(
        id="smolvlm",
        display_name="SmolVLM-256M",
        category="Vision Language Model",
        parameters="256M",
        use_cases=[
            "Low End GPU",
            "Laptop",
            "Edge Device",
            "Jetson",
            "Offline OCR",
        ],
        min_vram_gb=2.0,
        min_recommendation="2 GB VRAM",
        cpu_supported=True,
        is_default=False,
        hf_repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        description=(
            "Ultra-compact 256M vision-language model ideal for edge devices, "
            "laptops, Jetson boards, and offline deployments where memory is "
            "severely constrained."
        ),
        download_size_gb=0.5,
    ),

    # ── 6. PaddleOCR ───────────────────────────────────────────────────────
    ModelMetadata(
        id="paddleocr",
        display_name="PaddleOCR",
        category="Traditional OCR",
        parameters="Small",
        use_cases=[
            "CPU",
            "Server",
            "Fast OCR",
            "Fallback Engine",
        ],
        min_vram_gb=0.0,
        min_recommendation="CPU",
        cpu_supported=True,
        is_default=False,
        hf_repo_id="PaddlePaddle/PaddleOCR",
        description=(
            "Lightweight traditional OCR engine powered by PaddlePaddle. "
            "Runs entirely on CPU, making it the ideal fallback engine for "
            "servers and environments without a GPU."
        ),
        download_size_gb=0.3,
    ),
]

def _normalize_key(s: str) -> str:
    """Normalize string by removing casing, hyphens, underscores, and spaces."""
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


# Build fast O(1) lookup maps
_REGISTRY: Dict[str, ModelMetadata] = {m.id: m for m in _CATALOG}
_NORMALIZED_MAP: Dict[str, ModelMetadata] = {}
for m in _CATALOG:
    _NORMALIZED_MAP[_normalize_key(m.id)] = m
    _NORMALIZED_MAP[_normalize_key(m.display_name)] = m


# ---------------------------------------------------------------------------
# ModelRegistry — public interface
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Central read-only registry of all officially supported TextLens models.

    All access is through class-methods; no instantiation required.
    """

    # Expose the raw catalog for iteration tests
    _catalog: List[ModelMetadata] = _CATALOG

    @classmethod
    def all(cls) -> List[ModelMetadata]:
        """Return every registered model in catalog order."""
        return list(_CATALOG)

    @classmethod
    def supported_ids(cls) -> List[str]:
        """Return the list of supported model ID strings."""
        return [m.id for m in _CATALOG]

    @classmethod
    def get(cls, model_id: str) -> ModelMetadata:
        """Retrieve metadata for a specific model by its canonical ID or display name.

        Parameters
        ----------
        model_id : str
            The model slug or display name (e.g. ``"glm-ocr"``, ``"florence2"``, ``"Florence-2"``).

        Returns
        -------
        ModelMetadata
            The registered metadata for the requested model.

        Raises
        ------
        textlens.models.exceptions.UnknownModelError
            If *model_id* is not in the official catalog.
        """
        # Import here to avoid circular imports
        from textlens.models.exceptions import UnknownModelError

        meta = _REGISTRY.get(model_id) or _NORMALIZED_MAP.get(_normalize_key(model_id))
        if meta is None:
            raise UnknownModelError(model_id, cls.supported_ids())
        return meta

    @classmethod
    def is_registered(cls, model_id: str) -> bool:
        """Return True if *model_id* is an officially registered model."""
        return model_id in _REGISTRY or _normalize_key(model_id) in _NORMALIZED_MAP

    @classmethod
    def default(cls) -> ModelMetadata:
        """Return the default TextLens model (``glm-ocr``)."""
        for m in _CATALOG:
            if m.is_default:
                return m
        # Fallback: first entry (should never happen given catalog design)
        return _CATALOG[0]

    @classmethod
    def is_registered(cls, model_id: str) -> bool:
        """Return True if *model_id* is an officially registered model."""
        return model_id in _REGISTRY
