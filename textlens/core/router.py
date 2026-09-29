"""
textlens.core.router
────────────────────
Intelligent, explainable model routing.

For every page that needs OCR the router picks a primary model and an
ordered fallback chain from the registry:

1. **Hard constraints** remove models that cannot run here: missing runtime
   (torch, transformers…), no GPU for GPU-only models, VRAM/RAM below the
   spec minimum, weights missing in offline mode, remote models without a
   configured endpoint, license policy, latency / memory budgets.
2. **Scoring** ranks what remains using the profile's quality/speed weights,
   the page's difficulty signals (tables, formulas, blur, density) and
   whether weights are already installed.  Measured latencies from
   ``textlens benchmark`` replace the built-in speed tiers when available.
3. **Explanation**: every decision lists its reasons and why each other
   model was rejected — ``ocr.explain("scan.pdf")`` or ``--explain``.

Profiles
    ``edge``      tiny ONNX model, low memory, no generative models
    ``fast``      PP-OCR on CPU/GPU; reuse existing OCR layers
    ``balanced``  PP-OCR for simple pages, a document VLM for complex pages
                  or low-confidence results (when one is installed and fits)
    ``accurate``  document VLM first, PP-OCR as fallback
    ``document``  Markdown/structure-first (tables, formulas, headings)
    ``server``    like ``accurate``, preferring a remote inference server
    ``auto``      picks one of the above from the detected hardware
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from textlens.models.specs import (
    STATUS_CATALOG,
    STATUS_REMOTE,
    TASK_FORMULA,
    TASK_MARKDOWN,
    TASK_TABLE,
    TASK_TEXT,
    ModelSpec,
    all_specs,
    get_spec,
)

logger = logging.getLogger("textlens.router")


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    allow_generative: bool = False
    generative_for: str = "never"  # never | complex | always
    quality_weight: float = 0.5
    speed_weight: float = 0.5
    dpi: int = 200
    backend_options: Dict[str, Any] = field(default_factory=dict)
    fallback: bool = False
    accept_confidence: float = 0.80
    review_confidence: float = 0.60
    reuse_ocr_layer: bool = False
    auto_download_gb: float = 0.1
    prefer_remote: bool = False


PROFILES: Dict[str, Profile] = {
    "edge": Profile(
        "edge",
        "Smallest footprint: ONNX PP-OCR, low-memory runtime, no generative models.",
        quality_weight=0.1,
        speed_weight=1.0,
        dpi=150,
        # low_memory (no ORT arena) is opt-in: it saves RAM on <=1 GB boards
        # but makes detection ~3x slower (measured).
        backend_options={"det_limit": 960, "rec_batch": 4},
        accept_confidence=0.70,
        review_confidence=0.50,
        reuse_ocr_layer=True,
    ),
    "fast": Profile(
        "fast",
        "Fast local OCR with PP-OCR; reuses existing OCR layers in scanned PDFs.",
        quality_weight=0.3,
        speed_weight=1.0,
        dpi=200,
        backend_options={"det_limit": 1280},
        reuse_ocr_layer=True,
    ),
    "balanced": Profile(
        "balanced",
        "PP-OCR for simple pages; a document VLM for tables, formulas, hard or low-confidence pages.",
        allow_generative=True,
        generative_for="complex",
        quality_weight=0.6,
        speed_weight=0.5,
        dpi=200,
        backend_options={"det_limit": 1600},
        fallback=True,
        accept_confidence=0.85,
    ),
    "accurate": Profile(
        "accurate",
        "Highest-quality local model first; lighter models as fallback.",
        allow_generative=True,
        generative_for="always",
        quality_weight=1.0,
        speed_weight=0.1,
        dpi=220,
        backend_options={"det_limit": 2048},
        fallback=True,
        accept_confidence=0.90,
        review_confidence=0.75,
    ),
    "document": Profile(
        "document",
        "Structure-first: Markdown with headings, tables and formulas from document VLMs.",
        allow_generative=True,
        generative_for="always",
        quality_weight=1.0,
        speed_weight=0.2,
        dpi=200,
        backend_options={"det_limit": 1600},
        fallback=True,
    ),
    "server": Profile(
        "server",
        "Throughput-oriented: prefers a remote inference server (vLLM etc.) when configured.",
        allow_generative=True,
        generative_for="always",
        quality_weight=1.0,
        speed_weight=0.3,
        dpi=200,
        backend_options={"det_limit": 1600, "concurrency": 16},
        fallback=True,
        prefer_remote=True,
    ),
}

# Rough per-page latency (ms) by speed tier, used only until the machine has
# measurements from `textlens benchmark`.  GPU figures; CPU is slower.
_TIER_LATENCY_MS = {5: 400, 4: 1200, 3: 3000, 2: 8000, 1: 20000}


@dataclass
class PageSignals:
    """What the router knows about a page before OCR."""

    kind: str = "image"  # image | scanned | broken_encoding | vector_text | mixed | native
    reasons: List[str] = field(default_factory=list)
    likely_table: bool = False
    likely_formulas: bool = False
    difficulty: float = 0.3
    has_ocr_layer: bool = False
    task: str = TASK_TEXT

    @property
    def complex(self) -> bool:
        return self.likely_table or self.likely_formulas or self.difficulty >= 0.65 or self.task in (TASK_MARKDOWN, TASK_TABLE, TASK_FORMULA)


@dataclass
class RoutingDecision:
    model: str
    fallbacks: List[str]
    profile: str
    dpi: int
    device: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    accept_confidence: float = 0.8
    review_confidence: float = 0.6
    fallback_enabled: bool = False

    def explain(self) -> str:
        lines = [f"Selected: {self.model}  (profile: {self.profile}, render: {self.dpi} dpi, device: {self.device or 'auto'})"]
        lines += [f"  • {r}" for r in self.reasons]
        if self.fallbacks:
            lines.append(f"Fallback chain: {' → '.join(self.fallbacks)} (below {self.accept_confidence:.2f} confidence)")
        if self.rejected:
            lines.append("Not selected:")
            lines += [f"  – {m}: {why}" for m, why in self.rejected.items()]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "fallbacks": self.fallbacks,
            "profile": self.profile,
            "dpi": self.dpi,
            "device": self.device,
            "reasons": self.reasons,
            "rejected": self.rejected,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


def load_measurements() -> Dict[str, Dict[str, float]]:
    """Per-model measurements saved by ``textlens benchmark`` on this machine."""
    from textlens.config import get_settings

    path = get_settings().home / "measurements.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class Router:
    """Choose models for pages. Stateless apart from its configuration."""

    def __init__(
        self,
        profile: Optional[str] = None,
        *,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        endpoint: Optional[str] = None,
        fallback: Optional[bool] = None,
        fallback_models: Optional[Sequence[str]] = None,
        min_confidence: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        memory_budget_gb: Optional[float] = None,
        quality: Optional[str] = None,
        allow_download: bool = True,
        allowed_licenses: Optional[Sequence[str]] = None,
        commercial_only: bool = False,
        dpi: Optional[int] = None,
        backend_options: Optional[Dict[str, Any]] = None,
        system: Any = None,
    ) -> None:
        from textlens.config import get_settings

        self.settings = get_settings()
        self._system = system
        self.requested_profile = (profile or self.settings.profile or "auto").lower()
        if self.requested_profile not in PROFILES and self.requested_profile != "auto":
            from textlens.errors import ConfigurationError

            raise ConfigurationError(
                f"Unknown profile {profile!r}.", hint=f"Choose one of: auto, {', '.join(PROFILES)}."
            )
        if quality == "high" and self.requested_profile == "auto":
            self.requested_profile = "accurate"
        elif quality == "low" and self.requested_profile == "auto":
            self.requested_profile = "fast"
        self.explicit_model = model
        self.backend = backend
        self.device = device or self.settings.device
        self.endpoint = endpoint
        self.fallback_override = fallback
        self.fallback_models = list(fallback_models or [])
        self.min_confidence = min_confidence
        self.latency_budget_ms = latency_budget_ms
        self.memory_budget_gb = memory_budget_gb
        self.allow_download = allow_download and not self.settings.offline
        self.allowed_licenses = {l.lower() for l in allowed_licenses} if allowed_licenses else None
        self.commercial_only = commercial_only
        self.dpi = dpi
        self.backend_options = dict(backend_options or {})
        self._measurements: Optional[Dict[str, Dict[str, float]]] = None
        self._explicit_spec = self._resolve_explicit_spec()
        self.profile = PROFILES[self._resolve_profile()]

    # ── setup ────────────────────────────────────────────────────────────
    @property
    def system(self) -> Any:
        if self._system is None:
            from textlens.runtime.system import inspect_system

            self._system = inspect_system()
        return self._system

    def _resolve_explicit_spec(self) -> Optional[ModelSpec]:
        if not self.explicit_model:
            return None
        from textlens.models.specs import find_spec

        spec = find_spec(self.explicit_model)
        if spec is not None:
            if self.backend == "openai" and spec.backend != "openai":
                # Serve a local model through a remote endpoint instead.
                from dataclasses import replace

                spec = replace(spec, backend="openai", adapter="textlens.backends.openai_compat:OpenAICompatibleBackend", status=STATUS_REMOTE, requires=())
            return spec
        if self.backend == "openai":
            # Any model the server hosts, even if TextLens does not know it.
            return ModelSpec(
                id=self.explicit_model,
                display_name=self.explicit_model,
                family="remote",
                provider="remote endpoint",
                description="Model served by an OpenAI-compatible endpoint.",
                backend="openai",
                adapter="textlens.backends.openai_compat:OpenAICompatibleBackend",
                tasks=frozenset({TASK_TEXT, TASK_MARKDOWN, TASK_TABLE, TASK_FORMULA}),
                markdown=True,
                hf_repo_id=self.explicit_model,
                status=STATUS_REMOTE,
            )
        return get_spec(self.explicit_model)  # raises UnknownModelError

    def _resolve_profile(self) -> str:
        if self.requested_profile != "auto":
            return self.requested_profile
        if self._explicit_spec is not None:
            return "accurate" if self._explicit_spec.is_generative else "fast"
        sysinfo = self.system
        if sysinfo.is_edge and not sysinfo.torch_cuda:
            return "edge"
        # A generative model is worth routing to only if it can actually run.
        from textlens.models import artifacts

        for spec in all_specs():
            if spec.is_generative and spec.backend == "transformers" and artifacts.is_installed(spec):
                ok, _ = self._hard_constraints(spec, PROFILES["balanced"])
                if ok:
                    return "balanced"
        if self._remote_endpoint() and any(s.backend == "openai" for s in all_specs()):
            return "server"
        return "fast"

    def _remote_endpoint(self) -> Optional[str]:
        from textlens.backends.openai_compat import configured_endpoint

        return configured_endpoint(self.endpoint)

    # ── constraints & scoring ────────────────────────────────────────────
    def device_for(self, spec: ModelSpec) -> Optional[str]:
        if spec.backend == "openai":
            return None
        if self.device:
            return self.device
        if spec.backend == "transformers":
            s = self.system
            fits = not s.vram_gb or s.vram_gb >= spec.min_vram_gb
            # A model too large for the GPU but practical on CPU runs on CPU
            # rather than failing with out-of-memory.
            return "cuda" if (s.torch_cuda and fits) else "cpu"
        return None  # ONNX Runtime chooses the best available provider

    def _estimated_latency(self, spec: ModelSpec) -> float:
        if self._measurements is None:
            self._measurements = load_measurements()
        measured = self._measurements.get(spec.id, {}).get("latency_ms_p50")
        if measured:
            return float(measured)
        base = _TIER_LATENCY_MS.get(spec.speed_tier, 3000)
        if spec.backend == "transformers" and not self.system.torch_cuda:
            base *= 8
        elif spec.backend == "onnx" and not self.system.ort_cuda:
            base *= 2
        return base

    def _hard_constraints(self, spec: ModelSpec, profile: Profile, explicit: bool = False) -> Tuple[bool, str]:
        from textlens.backends.loader import install_hint, missing_requirements
        from textlens.models import artifacts

        if spec.status == STATUS_CATALOG:
            return False, "catalog entry (no bundled adapter)"
        if spec.backend == "openai" or spec.status == STATUS_REMOTE:
            if not self._remote_endpoint():
                return False, "remote model; no endpoint configured (set TEXTLENS_OPENAI_BASE_URL)"
            return True, ""
        if spec.is_generative and not profile.allow_generative and not explicit:
            return False, f"profile '{profile.name}' excludes generative models"
        missing = missing_requirements(spec)
        if missing:
            return False, f"needs {', '.join(missing)} ({install_hint(spec)})"
        sysinfo = self.system
        gpu_ok = sysinfo.torch_cuda if spec.backend == "transformers" else True
        device = (self.device or "").lower()
        if device.startswith("cpu"):
            gpu_ok = False
        if not spec.cpu and not gpu_ok:
            return False, "requires a CUDA GPU"
        if spec.backend == "transformers" and gpu_ok and sysinfo.vram_gb and sysinfo.vram_gb < spec.min_vram_gb:
            if not spec.cpu_practical:
                return False, f"needs {spec.min_vram_gb:g} GB VRAM (have {sysinfo.vram_gb:g} GB)"
        if spec.backend == "transformers" and not gpu_ok and not spec.cpu_practical and not explicit:
            return False, "too slow on CPU (no CUDA-enabled PyTorch)"
        if sysinfo.ram_gb and spec.min_ram_gb > sysinfo.ram_gb:
            return False, f"needs {spec.min_ram_gb:g} GB RAM (have {sysinfo.ram_gb:g} GB)"
        if self.memory_budget_gb is not None:
            need = spec.min_vram_gb if (gpu_ok and spec.min_vram_gb) else spec.min_ram_gb
            if need > self.memory_budget_gb:
                return False, f"needs ~{need:g} GB, over the {self.memory_budget_gb:g} GB memory budget"
        if self.latency_budget_ms is not None:
            est = self._estimated_latency(spec)
            if est > self.latency_budget_ms:
                return False, f"estimated {est:.0f} ms/page exceeds the {self.latency_budget_ms:.0f} ms budget"
        if self.allowed_licenses is not None and spec.license.lower() not in self.allowed_licenses:
            return False, f"license {spec.license} not in the allowed list"
        if self.commercial_only and spec.commercial_use != "yes":
            return False, f"commercial use {spec.commercial_use} ({spec.license})"
        if not artifacts.is_installed(spec):
            size = spec.download_size_gb or 0.0
            if self.settings.offline:
                return False, "not installed (offline mode)"
            if not explicit and not (self.allow_download and size <= profile.auto_download_gb):
                return False, f"not installed (~{size:g} GB); run `textlens models install {spec.id}`"
        return True, ""

    def _score(self, spec: ModelSpec, profile: Profile, signals: PageSignals) -> float:
        from textlens.models import artifacts

        speed = spec.speed_tier
        measured = (self._measurements or {}).get(spec.id, {}).get("latency_ms_p50")
        if measured:
            speed = 5 if measured < 500 else 4 if measured < 1500 else 3 if measured < 4000 else 2 if measured < 10000 else 1
        score = profile.quality_weight * spec.quality_tier + profile.speed_weight * speed
        if spec.is_generative:
            if profile.generative_for == "always" or signals.task in (TASK_MARKDOWN, TASK_TABLE, TASK_FORMULA):
                score += 2.0
            elif profile.generative_for == "complex":
                score += 2.0 if signals.complex else -2.0
        if signals.likely_table and spec.supports(TASK_TABLE):
            score += 0.5
        if signals.likely_formulas and spec.supports(TASK_FORMULA):
            score += 0.5
        if profile.prefer_remote and spec.backend == "openai":
            score += 1.5
        if artifacts.is_installed(spec) or spec.backend == "openai":
            score += 0.5
        return score

    # ── public API ───────────────────────────────────────────────────────
    def route(self, signals: Optional[PageSignals] = None) -> RoutingDecision:
        from textlens.errors import RoutingError

        signals = signals or PageSignals()
        profile = self.profile
        accept = self.min_confidence if self.min_confidence is not None else profile.accept_confidence
        fallback_enabled = profile.fallback if self.fallback_override is None else self.fallback_override
        dpi = self.dpi or profile.dpi
        rejected: Dict[str, str] = {}

        if self._explicit_spec is not None:
            spec = self._explicit_spec
            ok, why = self._hard_constraints(spec, profile, explicit=True)
            if not ok and spec.backend == "openai":
                raise RoutingError(f"Cannot use {spec.id}: {why}", hint="Pass endpoint='http://host:8000/v1'.")
            fallbacks = [m for m in self.fallback_models if m != spec.id]
            reasons = [f"explicitly requested model {spec.id}"]
            if not ok:
                reasons.append(f"warning: {why}")
            return RoutingDecision(
                model=spec.id,
                fallbacks=fallbacks if (fallback_enabled or self.fallback_models) else [],
                profile=profile.name,
                dpi=dpi,
                device=self.device_for(spec),
                options=self._options_for(spec, profile),
                reasons=reasons,
                accept_confidence=accept,
                review_confidence=profile.review_confidence,
                fallback_enabled=bool(fallback_enabled or self.fallback_models),
            )

        scored: List[Tuple[float, ModelSpec]] = []
        for spec in all_specs():
            if not spec.supports(TASK_TEXT) and not spec.supports(signals.task):
                continue
            ok, why = self._hard_constraints(spec, profile)
            if not ok:
                rejected[spec.id] = why
                continue
            scored.append((self._score(spec, profile, signals), spec))
        if not scored:
            raise RoutingError(
                "No OCR model can run with the current profile, hardware and installed packages.",
                hint="Run `textlens doctor` for a diagnosis, or `textlens setup` to install a model.",
                rejected=rejected,
            )
        scored.sort(key=lambda t: t[0], reverse=True)
        primary = scored[0][1]
        chain = [s.id for _, s in scored[1:]] if fallback_enabled else []
        if self.fallback_models:
            chain = [m for m in self.fallback_models if m != primary.id]
        reasons = [f"profile '{profile.name}': {profile.description}"]
        if self.requested_profile == "auto":
            reasons.append(f"auto profile resolved to '{profile.name}' for {self._hardware_summary()}")
        if signals.reasons:
            reasons.append(f"page needs OCR because: {', '.join(signals.reasons)}")
        if signals.complex:
            what = [w for w, on in (("tables", signals.likely_table), ("formulas", signals.likely_formulas), (f"difficulty {signals.difficulty:.2f}", signals.difficulty >= 0.65)) if on]
            reasons.append("complex page (" + (", ".join(what) or f"task {signals.task}") + ")")
        reasons.append(f"{primary.display_name}: quality tier {primary.quality_tier}, speed tier {primary.speed_tier}, license {primary.license}")
        return RoutingDecision(
            model=primary.id,
            fallbacks=chain,
            profile=profile.name,
            dpi=dpi,
            device=self.device_for(primary),
            options=self._options_for(primary, profile),
            reasons=reasons,
            rejected=rejected,
            scores={s.id: sc for sc, s in scored},
            accept_confidence=accept,
            review_confidence=profile.review_confidence,
            fallback_enabled=fallback_enabled,
        )

    def _options_for(self, spec: ModelSpec, profile: Profile) -> Dict[str, Any]:
        opts: Dict[str, Any] = {}
        if spec.backend == "onnx":
            opts.update({k: v for k, v in profile.backend_options.items() if k in ("det_limit", "low_memory", "rec_batch", "threads")})
        if spec.backend == "openai":
            if self.endpoint:
                opts["endpoint"] = self.endpoint
            if "concurrency" in profile.backend_options:
                opts["concurrency"] = profile.backend_options["concurrency"]
        opts.update(self.backend_options)
        return opts

    def _hardware_summary(self) -> str:
        s = self.system
        gpu = s.hardware.primary_gpu_name
        if gpu:
            return f"{gpu} ({s.vram_gb:g} GB VRAM, torch CUDA: {'yes' if s.torch_cuda else 'no'})"
        return f"CPU-only {s.platform.device_class} ({s.ram_gb:.0f} GB RAM)"

    def options_for(self, model_id: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Device and backend options for a (fallback) model id."""
        spec = get_spec(model_id) if (self._explicit_spec is None or self._explicit_spec.id != model_id) else self._explicit_spec
        return self.device_for(spec), self._options_for(spec, self.profile)

    def spec_for(self, model_id: str) -> ModelSpec:
        if self._explicit_spec is not None and self._explicit_spec.id == model_id:
            return self._explicit_spec
        return get_spec(model_id)
