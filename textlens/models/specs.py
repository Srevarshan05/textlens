"""
textlens.models.specs
─────────────────────
The TextLens model catalog — single source of truth for every engine.

Each :class:`ModelSpec` declares what a model can do (tasks, outputs,
languages), what it needs (backend runtime, CPU/GPU, VRAM, RAM), where its
weights come from (Hugging Face repo or pinned URL artifacts with SHA-256),
and its license.  The router, doctor, CLI and server all read from here;
no model-specific logic lives in the core.

Honesty rules for this file
    * Licenses were read from each model's Hugging Face card metadata.
    * ``quality_tier`` / ``speed_tier`` are **TextLens heuristics** used for
      routing, not benchmark results.  Measure on your own data with
      ``textlens benchmark`` — measured numbers override these tiers.
    * No official benchmark figures are copied here; see each model card.

Third-party models can register specs at runtime with
:func:`register_spec` or through the ``textlens.models`` entry-point group.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

logger = logging.getLogger("textlens.models.specs")

# Tasks
TASK_TEXT = "text"
TASK_LAYOUT = "layout"
TASK_TABLE = "table"
TASK_FORMULA = "formula"
TASK_HANDWRITING = "handwriting"
TASK_MARKDOWN = "markdown"
TASK_EXTRACTION = "extraction"
TASK_CHART = "chart"
TASK_PLATE = "plate"

# Status
STATUS_STABLE = "stable"
STATUS_EXPERIMENTAL = "experimental"
STATUS_REMOTE = "remote"  # runnable through an OpenAI-compatible server
STATUS_CATALOG = "catalog"  # documented, no bundled adapter


@dataclasses.dataclass(frozen=True)
class Artifact:
    """A pinned downloadable file (URL + SHA-256)."""

    filename: str
    url: str
    sha256: str
    size_bytes: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    family: str
    provider: str
    description: str
    backend: str  # onnx | transformers | openai | external
    adapter: Optional[str]  # "module:Class" of the backend adapter
    tasks: FrozenSet[str]
    parameters: Optional[str] = None
    languages: Tuple[str, ...] = ("en",)
    # Outputs
    boxes: bool = False
    confidence: bool = False
    markdown: bool = False
    # Hardware
    cpu: bool = True
    cuda: bool = False
    mps: bool = False
    edge: bool = False
    cpu_practical: bool = False  # usable speed on CPU (not just "it runs")
    min_vram_gb: float = 0.0
    recommended_vram_gb: float = 0.0
    min_ram_gb: float = 1.0
    quantization: Tuple[str, ...] = ()
    # Runtime requirements: importable module names (checked without import)
    requires: Tuple[str, ...] = ()
    extra: Optional[str] = None  # pip extra that installs `requires`
    # Weights
    hf_repo_id: Optional[str] = None
    revision: Optional[str] = None
    artifacts: Tuple[Artifact, ...] = ()
    download_size_gb: Optional[float] = None
    # Governance
    license: str = "unknown"
    license_url: Optional[str] = None
    commercial_use: str = "unknown"  # yes | restricted | no | unknown
    license_notes: Optional[str] = None
    source_url: Optional[str] = None
    # Routing heuristics (1 = low, 5 = high); overridden by measurements
    quality_tier: int = 3
    speed_tier: int = 3
    status: str = STATUS_STABLE
    limitations: Tuple[str, ...] = ()
    prompts: Mapping[str, str] = dataclasses.field(default_factory=dict)
    use_cases: Tuple[str, ...] = ()  # descriptive tags shown in catalogs
    is_default: bool = False
    aliases: Tuple[str, ...] = ()

    # ── helpers ──────────────────────────────────────────────────────────
    def supports(self, task: str) -> bool:
        return task in self.tasks

    @property
    def is_generative(self) -> bool:
        return self.backend in ("transformers", "openai") and TASK_MARKDOWN in self.tasks

    @property
    def runnable_locally(self) -> bool:
        return self.status in (STATUS_STABLE, STATUS_EXPERIMENTAL) and self.adapter is not None

    def to_dict(self) -> Dict[str, object]:
        data = dataclasses.asdict(self)
        data["tasks"] = sorted(self.tasks)
        data["prompts"] = dict(self.prompts)
        return data

    def to_metadata(self):  # -> ModelMetadata (legacy view)
        from textlens.models.metadata import ModelMetadata

        uses = {
            TASK_TEXT: "General OCR",
            TASK_LAYOUT: "Layout",
            TASK_TABLE: "Tables",
            TASK_FORMULA: "Formula Recognition",
            TASK_HANDWRITING: "Handwriting",
            TASK_MARKDOWN: "Markdown Export",
            TASK_EXTRACTION: "Information Extraction",
            TASK_CHART: "Charts",
            TASK_PLATE: "License Plates",
        }
        use_cases = list(self.use_cases) or [uses[t] for t in (TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_LAYOUT, TASK_MARKDOWN, TASK_EXTRACTION, TASK_CHART, TASK_HANDWRITING, TASK_PLATE) if t in self.tasks]
        if self.edge:
            use_cases.append("Edge Devices")
        if len(self.languages) > 3:
            use_cases.append("Multilingual OCR")
        if self.min_vram_gb > 0:
            rec = f"{self.min_vram_gb:g} GB VRAM"
        else:
            rec = "CPU (any)" if self.cpu else "GPU"
        return ModelMetadata(
            id=self.id,
            display_name=self.display_name,
            category="OCR" if not self.is_generative else "Vision Language Model",
            parameters=self.parameters or "n/a",
            use_cases=use_cases,
            min_vram_gb=self.min_vram_gb,
            min_recommendation=rec,
            cpu_supported=self.cpu,
            is_default=self.is_default,
            hf_repo_id=self.hf_repo_id or (self.artifacts[0].url.rsplit("/", 1)[0] if self.artifacts else ""),
            description=self.description,
            download_size_gb=self.download_size_gb,
        )


_OAR = "https://github.com/GreatV/oar-ocr/releases/download/v0.7.0"

_BUILTIN: List[ModelSpec] = [
    # ── Edge / default: PP-OCRv6 Small on ONNX Runtime ──────────────────────
    ModelSpec(
        id="ppocrv6-small",
        display_name="PP-OCRv6 Small",
        family="PP-OCR",
        provider="PaddlePaddle (ONNX export by oar-ocr)",
        description=(
            "Lightweight text detection + recognition (DBNet + SVTR-style CTC). Runs on any CPU through "
            "ONNX Runtime, including Raspberry Pi and Jetson; returns line boxes and confidences."
        ),
        backend="onnx",
        adapter="textlens.backends.onnx.ppocr:PPOCRBackend",
        tasks=frozenset({TASK_TEXT}),
        parameters="~31 MB (det + rec)",
        languages=("en", "zh", "ja", "latin"),
        boxes=True,
        confidence=True,
        cpu=True,
        cuda=True,
        edge=True,
        cpu_practical=True,
        min_ram_gb=0.5,
        requires=("onnxruntime", "numpy"),
        extra=None,
        artifacts=(
            Artifact("pp-ocrv6_small_det.onnx", f"{_OAR}/pp-ocrv6_small_det.onnx", "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e", 9880512),
            Artifact("pp-ocrv6_small_rec.onnx", f"{_OAR}/pp-ocrv6_small_rec.onnx", "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634", 21159378),
            Artifact("ppocrv6_dict.txt", f"{_OAR}/ppocrv6_dict.txt", "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d", 74947),
        ),
        revision="oar-ocr-v0.7.0",
        download_size_gb=0.031,
        license="apache-2.0",
        license_url="https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE",
        commercial_use="yes",
        source_url="https://github.com/GreatV/oar-ocr/releases/tag/v0.7.0",
        quality_tier=3,
        speed_tier=5,
        limitations=(
            "Plain text lines only: no table structure, formulas or Markdown.",
            "No text-line orientation classifier: upside-down text is not corrected.",
            "Handwriting accuracy is limited.",
        ),
        use_cases=("General OCR", "Scanned Documents", "Receipts", "Edge Devices", "Raspberry Pi", "Jetson", "Offline OCR"),
        is_default=True,
        aliases=("ppocr", "pp-ocrv6", "ppocrv6"),
    ),
    # ── Local document VLMs (PyTorch / Transformers) ────────────────────────
    ModelSpec(
        id="glm-ocr",
        display_name="GLM-OCR",
        family="GLM",
        provider="Z.ai (zai-org)",
        description=(
            "Document OCR VLM for text, tables and formulas with Markdown output. "
            "Strong default for complex pages on a mid-range GPU."
        ),
        backend="transformers",
        adapter="textlens.backends.transformers_vlm:GLMOCRAdapter",
        tasks=frozenset({TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_MARKDOWN, TASK_LAYOUT, TASK_EXTRACTION}),
        parameters="0.9B (1.33B tensors incl. vision)",
        languages=("en", "zh", "multilingual"),
        markdown=True,
        cpu=True,
        cuda=True,
        min_vram_gb=4.0,
        recommended_vram_gb=6.0,
        min_ram_gb=8.0,
        quantization=("bf16", "fp16"),
        requires=("torch", "transformers"),
        extra="gpu",
        hf_repo_id="zai-org/GLM-OCR",
        download_size_gb=2.7,
        license="mit",
        commercial_use="yes",
        source_url="https://huggingface.co/zai-org/GLM-OCR",
        quality_tier=4,
        speed_tier=2,
        limitations=(
            "Requires a transformers release that ships GlmOcrForConditionalGeneration (5.x).",
            "Generative: no per-region boxes; confidence is a token-probability proxy.",
            "CPU inference is very slow.",
        ),
        prompts={"text": "Text Recognition:", "table": "Table Recognition:", "formula": "Formula Recognition:"},
        use_cases=("General OCR", "Invoices", "Books", "Research Papers", "Forms", "Tables", "Formula Recognition", "Markdown Export"),
        aliases=("glm", "glmocr"),
    ),
    ModelSpec(
        id="lighton-ocr",
        display_name="LightOnOCR-2",
        family="LightOnOCR",
        provider="LightOn",
        description="1B end-to-end OCR VLM tuned for scientific/academic PDFs; Markdown output.",
        backend="transformers",
        adapter="textlens.backends.transformers_vlm:LightOnOCRAdapter",
        tasks=frozenset({TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_MARKDOWN, TASK_LAYOUT}),
        parameters="1B",
        languages=("en", "fr", "de", "es", "it", "multilingual"),
        markdown=True,
        cpu=True,
        cuda=True,
        min_vram_gb=6.0,
        recommended_vram_gb=8.0,
        min_ram_gb=8.0,
        quantization=("bf16",),
        requires=("torch", "transformers"),
        extra="gpu",
        hf_repo_id="lightonai/LightOnOCR-2-1B",
        download_size_gb=2.0,
        license="apache-2.0",
        commercial_use="yes",
        source_url="https://huggingface.co/lightonai/LightOnOCR-2-1B",
        quality_tier=4,
        speed_tier=2,
        limitations=("Requires transformers >= 5.0.", "Fixed OCR prompt; custom prompts are ignored."),
        use_cases=("Scientific Documents", "Academic Papers", "Multilingual OCR", "PDF Parsing"),
        aliases=("lighton", "lightonocr"),
    ),
    ModelSpec(
        id="hunyuan-ocr",
        display_name="HunyuanOCR",
        family="Hunyuan",
        provider="Tencent",
        description="~1B OCR VLM for complex layouts, charts and information extraction (Markdown + HTML tables).",
        backend="transformers",
        adapter="textlens.backends.transformers_vlm:HunyuanOCRAdapter",
        tasks=frozenset({TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_MARKDOWN, TASK_LAYOUT, TASK_CHART, TASK_EXTRACTION}),
        parameters="~1B",
        languages=("zh", "en", "multilingual"),
        markdown=True,
        cpu=False,
        cuda=True,
        min_vram_gb=6.0,
        recommended_vram_gb=8.0,
        min_ram_gb=12.0,
        requires=("torch", "transformers"),
        extra="gpu",
        hf_repo_id="tencent/HunyuanOCR",
        download_size_gb=2.0,
        license="tencent-hunyuan-community",
        commercial_use="restricted",
        license_notes="Custom Tencent license with territorial and usage restrictions — review before use.",
        source_url="https://huggingface.co/tencent/HunyuanOCR",
        quality_tier=4,
        speed_tier=2,
        limitations=("Requires transformers >= 5.0 and trust_remote_code.", "CPU inference is not practical."),
        use_cases=("Enterprise Documents", "Charts", "Tables", "Complex Layouts", "Information Extraction"),
        aliases=("hunyuan", "hunyuanocr"),
    ),
    ModelSpec(
        id="smolvlm",
        display_name="SmolVLM-256M",
        family="SmolVLM",
        provider="Hugging Face",
        description="Tiny general VLM usable for rough OCR on laptops; not a dedicated OCR model.",
        backend="transformers",
        adapter="textlens.backends.transformers_vlm:SmolVLMAdapter",
        tasks=frozenset({TASK_TEXT, TASK_MARKDOWN}),
        parameters="256M",
        languages=("en",),
        markdown=True,
        cpu=True,
        cuda=True,
        cpu_practical=True,
        min_vram_gb=2.0,
        min_ram_gb=4.0,
        requires=("torch", "transformers"),
        extra="gpu",
        hf_repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        download_size_gb=0.5,
        license="apache-2.0",
        commercial_use="yes",
        source_url="https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct",
        quality_tier=1,
        speed_tier=3,
        limitations=("General-purpose VLM: hallucinates on dense documents; prefer ppocrv6-small for plain text.",),
        prompts={"text": "Extract all text from this image:"},
        use_cases=("Low End GPU", "Laptop", "Edge Device", "Jetson", "Offline OCR"),
        aliases=("smol", "smolvlm-256m"),
    ),
    # ── Served models (OpenAI-compatible endpoint, e.g. vLLM) ────────────────
    ModelSpec(
        id="paddleocr-vl",
        display_name="PaddleOCR-VL",
        family="PaddleOCR",
        provider="PaddlePaddle",
        description="0.9B document parsing VLM (text, tables, formulas, charts; 100+ languages). Serve with vLLM or PaddleOCR.",
        backend="openai",
        adapter="textlens.backends.openai_compat:OpenAICompatibleBackend",
        tasks=frozenset({TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_MARKDOWN, TASK_LAYOUT, TASK_CHART}),
        parameters="0.9B",
        languages=("multilingual",),
        markdown=True,
        cpu=False,
        cuda=True,
        min_vram_gb=4.0,
        hf_repo_id="PaddlePaddle/PaddleOCR-VL",
        license="apache-2.0",
        commercial_use="yes",
        source_url="https://huggingface.co/PaddlePaddle/PaddleOCR-VL",
        quality_tier=5,
        speed_tier=3,
        status=STATUS_REMOTE,
        limitations=("Point TextLens at a server hosting it: OCR(model='paddleocr-vl', endpoint='http://host:8000/v1').",),
        prompts={"text": "OCR:", "table": "Table Recognition:", "formula": "Formula Recognition:", "chart": "Chart Recognition:"},
    ),
    ModelSpec(
        id="deepseek-ocr-2",
        display_name="DeepSeek-OCR-2",
        family="DeepSeek-OCR",
        provider="DeepSeek",
        description="3B vision-token-compression OCR model; strong on long, dense documents. Serve with vLLM.",
        backend="openai",
        adapter="textlens.backends.openai_compat:OpenAICompatibleBackend",
        tasks=frozenset({TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_MARKDOWN, TASK_LAYOUT, TASK_CHART}),
        parameters="3.4B",
        languages=("multilingual",),
        markdown=True,
        cpu=False,
        cuda=True,
        min_vram_gb=10.0,
        recommended_vram_gb=16.0,
        hf_repo_id="deepseek-ai/DeepSeek-OCR-2",
        license="apache-2.0",
        commercial_use="yes",
        source_url="https://huggingface.co/deepseek-ai/DeepSeek-OCR-2",
        quality_tier=5,
        speed_tier=2,
        status=STATUS_REMOTE,
        prompts={"text": "<image>\nFree OCR.", "markdown": "<image>\n<|grounding|>Convert the document to markdown."},
        limitations=("Needs a GPU server; not practical on laptops.",),
    ),
    ModelSpec(
        id="deepseek-ocr",
        display_name="DeepSeek-OCR",
        family="DeepSeek-OCR",
        provider="DeepSeek",
        description="3B optical-compression OCR model (v1). Serve with vLLM.",
        backend="openai",
        adapter="textlens.backends.openai_compat:OpenAICompatibleBackend",
        tasks=frozenset({TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_MARKDOWN, TASK_LAYOUT}),
        parameters="3.3B",
        languages=("multilingual",),
        markdown=True,
        cpu=False,
        cuda=True,
        min_vram_gb=10.0,
        hf_repo_id="deepseek-ai/DeepSeek-OCR",
        license="mit",
        commercial_use="yes",
        source_url="https://huggingface.co/deepseek-ai/DeepSeek-OCR",
        quality_tier=4,
        speed_tier=2,
        status=STATUS_REMOTE,
        prompts={"text": "<image>\nFree OCR.", "markdown": "<image>\n<|grounding|>Convert the document to markdown."},
    ),
    # ── Catalog-only entries (documented, license-sensitive) ────────────────
    ModelSpec(
        id="mineru2.5",
        display_name="MinerU2.5",
        family="MinerU",
        provider="OpenDataLab",
        description="1.2B document parsing VLM (layout + recognition). Catalog entry only.",
        backend="external",
        adapter=None,
        tasks=frozenset({TASK_TEXT, TASK_TABLE, TASK_FORMULA, TASK_MARKDOWN, TASK_LAYOUT}),
        parameters="1.2B",
        languages=("multilingual",),
        markdown=True,
        cpu=False,
        cuda=True,
        min_vram_gb=8.0,
        hf_repo_id="opendatalab/MinerU2.5-2509-1.2B",
        license="agpl-3.0",
        commercial_use="restricted",
        license_notes="AGPL-3.0: network use triggers source-disclosure obligations. Not bundled by TextLens.",
        source_url="https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B",
        quality_tier=5,
        speed_tier=2,
        status=STATUS_CATALOG,
    ),
    ModelSpec(
        id="surya",
        display_name="Surya",
        family="Surya",
        provider="Datalab",
        description="OCR + layout + reading order toolkit (90+ languages). Catalog entry only.",
        backend="external",
        adapter=None,
        tasks=frozenset({TASK_TEXT, TASK_LAYOUT, TASK_TABLE}),
        languages=("multilingual",),
        boxes=True,
        confidence=True,
        cpu=True,
        cuda=True,
        min_vram_gb=4.0,
        license="gpl-3.0 (code) / modified OpenRAIL-M (weights)",
        commercial_use="restricted",
        license_notes="GPL-3.0 code; weights carry revenue-based commercial limits. Verify before commercial use.",
        source_url="https://github.com/datalab-to/surya",
        quality_tier=4,
        speed_tier=3,
        status=STATUS_CATALOG,
    ),
]


# ── registry storage ─────────────────────────────────────────────────────────

_lock = threading.RLock()
_SPECS: Dict[str, ModelSpec] = {}
_ALIASES: Dict[str, str] = {}
_plugins_loaded = False


def _norm(key: str) -> str:
    return key.lower().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")


def register_spec(spec: ModelSpec, replace: bool = False) -> None:
    """Add a model to the catalog (third-party adapters use this)."""
    with _lock:
        if spec.id in _SPECS and not replace:
            raise ValueError(f"Model id {spec.id!r} is already registered")
        _SPECS[spec.id] = spec
        for key in (spec.id, spec.display_name, spec.hf_repo_id or "", *spec.aliases):
            if key:
                _ALIASES[_norm(key)] = spec.id


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group = eps.select(group="textlens.models") if hasattr(eps, "select") else eps.get("textlens.models", [])
        for ep in group:
            try:
                obj = ep.load()
                specs = obj() if callable(obj) else obj
                for spec in specs if isinstance(specs, (list, tuple)) else [specs]:
                    register_spec(spec, replace=True)
            except Exception as exc:  # a broken plugin must not break TextLens
                logger.warning("Ignoring model plugin %s: %s", ep.name, exc)
    except Exception:
        pass


for _spec in _BUILTIN:
    register_spec(_spec)


def all_specs() -> List[ModelSpec]:
    _load_plugins()
    with _lock:
        return list(_SPECS.values())


def get_spec(model_id: str) -> ModelSpec:
    from textlens.errors import UnknownModelError

    _load_plugins()
    with _lock:
        if model_id in _SPECS:
            return _SPECS[model_id]
        target = _ALIASES.get(_norm(model_id))
        if target:
            return _SPECS[target]
        raise UnknownModelError(model_id, list(_SPECS))


def find_spec(model_id: str) -> Optional[ModelSpec]:
    try:
        return get_spec(model_id)
    except Exception:
        return None


def default_spec() -> ModelSpec:
    for s in all_specs():
        if s.is_default:
            return s
    return all_specs()[0]


def search_specs(query: str) -> List[ModelSpec]:
    """Case-insensitive search over ids, names, tasks, languages and descriptions."""
    q = query.lower().strip()
    out = []
    for s in all_specs():
        hay = " ".join(
            [s.id, s.display_name, s.family, s.provider, s.description, s.license, s.backend, *s.tasks, *s.languages, *s.aliases]
        ).lower()
        if all(tok in hay for tok in q.split()):
            out.append(s)
    return out
