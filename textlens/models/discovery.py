"""Live Hugging Face model discovery with hardware-aware recommendations.

The official TextLens registry remains the source of supported inference
backends. This module discovers *candidate* OCR/VLM repositories so developers
can research models that fit their hardware; discovered repositories are not
automatically treated as TextLens-supported backends.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, List, Optional

from textlens.models.hardware import HardwareProfile, inspect_hardware
from textlens.models.registry import ModelRegistry


_DISCOVERY_CACHE_TTL_SECONDS = 15 * 60


def _cache_path() -> Path:
    """Return TextLens' per-user cache location without touching a repo."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "TextLens" / "discovery-cache.json"
    return Path.home() / ".cache" / "textlens" / "discovery-cache.json"


def _cache_key(
    search: str,
    use_case: Optional[str],
    limit: int,
    compatible_only: bool,
    include_unknown: bool,
    profile: HardwareProfile,
) -> str:
    """Create a stable cache key that includes hardware-sensitive options."""
    payload = json.dumps(
        {
            "search": search.casefold(),
            "use_case": (use_case or "").casefold(),
            "limit": limit,
            "compatible_only": compatible_only,
            "include_unknown": include_unknown,
            "cuda": profile.cuda_available,
            "vram": profile.primary_vram_gb,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(key: str) -> Optional[List[DiscoveredModel]]:
    """Read a fresh cached discovery response, ignoring corrupt cache files."""
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
        entry = data.get(key)
        if not entry or time.time() - float(entry["saved_at"]) > _DISCOVERY_CACHE_TTL_SECONDS:
            return None
        return [DiscoveredModel(**item) for item in entry["models"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(key: str, models: List[DiscoveredModel]) -> None:
    """Persist discovery results atomically; caching must never break the CLI."""
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[key] = {
            "saved_at": time.time(),
            "models": [dataclasses.asdict(item) for item in models],
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, default=str), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


@dataclasses.dataclass(frozen=True)
class DiscoveredModel:
    """A Hugging Face model candidate enriched with an estimated VRAM tier."""

    repo_id: str
    pipeline_tag: Optional[str]
    tags: List[str]
    downloads: int
    likes: int
    parameter_count_b: Optional[float]
    estimated_vram_gb: Optional[float]
    compatibility: str
    use_case_signals: List[str]


def _parameter_count_billion(model: Any) -> Optional[float]:
    """Read a parameter count from Hub metadata or a repository-name fallback."""
    metadata = getattr(model, "safetensors", None)
    if isinstance(metadata, dict):
        total = metadata.get("total")
        if isinstance(total, (int, float)) and total > 0:
            return round(float(total) / 1_000_000_000, 3)

    card_data = getattr(model, "cardData", None) or {}
    config = getattr(model, "config", None) or {}
    for source in (card_data, config):
        if not isinstance(source, dict):
            continue
        for key in ("parameters", "parameter_count", "num_parameters", "params"):
            value = source.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return round(float(value) / 1_000_000_000, 3)
            if isinstance(value, str):
                parsed = _parse_parameter_label(value)
                if parsed is not None:
                    return parsed

    return _parse_parameter_label(str(getattr(model, "modelId", "")))


def _parse_parameter_label(value: str) -> Optional[float]:
    """Parse labels such as ``0.9B``, ``1B``, or ``256M`` into billions."""
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([bBmMkK])(?![a-zA-Z])", value)
    if not match:
        return None
    amount = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"b": 1.0, "m": 0.001, "k": 0.000001}[suffix]
    return round(amount * multiplier, 3)


def _estimate_vram_gb(parameter_count_b: Optional[float]) -> Optional[float]:
    """Return a conservative fp16 inference estimate, or ``None`` if unknown.

    The estimate includes model weights plus a modest runtime/activation margin.
    It is guidance only: image size, context length, quantisation, and runtime
    configuration can materially change real memory usage.
    """
    if parameter_count_b is None:
        return None
    return max(2.0, float(math.ceil(parameter_count_b * 2.5)))


def _official_metadata(repo_id: str) -> Any:
    """Return static TextLens metadata when a live result is supported."""
    target = repo_id.casefold()
    return next(
        (item for item in ModelRegistry.all() if item.hf_repo_id.casefold() == target),
        None,
    )


def _safetensors_parameter_count(api: Any, repo_id: str) -> Optional[float]:
    """Read an exact parameter count from published safetensors metadata.

    The Hub does not expose parameter counts for every repository in its search
    response. Safetensors repositories can publish a tensor-level count, which
    is the most reliable generic source for the hardware advisor.
    """
    try:
        metadata = api.get_safetensors_metadata(repo_id, timeout=2)
        counts = getattr(metadata, "parameter_count", None)
        if isinstance(counts, dict):
            total = sum(value for value in counts.values() if isinstance(value, int))
            if total > 0:
                return round(total / 1_000_000_000, 3)
    except Exception:
        # Many repositories use non-safetensors weights or do not publish
        # enough metadata. They are handled by the explicit unknown filter.
        pass
    return None


def _published_parameter_counts(api: Any, models: List[Any]) -> dict[str, float]:
    """Fetch missing published parameter metadata concurrently and safely."""
    missing = [
        str(getattr(model, "modelId", ""))
        for model in models
        if _official_metadata(str(getattr(model, "modelId", ""))) is None
        and _parameter_count_billion(model) is None
    ]
    counts: dict[str, float] = {}
    if not missing:
        return counts

    # Network calls are bounded: discovery remains responsive while retrieving
    # the extra metadata needed for an honest hardware recommendation.
    with ThreadPoolExecutor(max_workers=min(4, len(missing))) as executor:
        futures = {
            executor.submit(_safetensors_parameter_count, api, repo_id): repo_id
            for repo_id in missing
        }
        for future in as_completed(futures):
            value = future.result()
            if value is not None:
                counts[futures[future]] = value
    return counts


def _compatibility(profile: HardwareProfile, required_vram_gb: Optional[float]) -> str:
    """Classify a model candidate against locally detected hardware."""
    if required_vram_gb is None:
        return "VRAM not published"
    if not profile.cuda_available:
        return "CPU fallback (slow)"
    if profile.primary_vram_gb >= required_vram_gb:
        return "Compatible"
    return f"Needs ~{required_vram_gb:.0f} GB VRAM"


def _use_case_signals(model: Any, requested_use_case: Optional[str]) -> List[str]:
    """Build readable use-case tags from Hugging Face metadata."""
    raw_tags = [str(tag).lower() for tag in (getattr(model, "tags", None) or [])]
    repo_id = str(getattr(model, "modelId", "")).lower()
    haystack = " ".join([repo_id, *raw_tags])
    signals: List[str] = []
    mapping = {
        "OCR": ("ocr", "image-to-text"),
        "Documents": ("document", "layout", "pdf"),
        "Tables": ("table", "chart"),
        "Handwriting": ("handwrit",),
        "Multilingual": ("multilingual", "language"),
        "Vision-language": ("vision", "vlm"),
    }
    for label, terms in mapping.items():
        if any(term in haystack for term in terms):
            signals.append(label)
    if requested_use_case:
        signals.insert(0, requested_use_case.title())
    return signals or ["OCR/VLM candidate"]


def discover_models(
    search: str = "ocr",
    use_case: Optional[str] = None,
    limit: int = 12,
    compatible_only: bool = False,
    include_unknown: bool = False,
    refresh: bool = False,
    use_cache: bool = False,
    profile: Optional[HardwareProfile] = None,
) -> List[DiscoveredModel]:
    """Discover OCR/VLM candidates from the Hugging Face Hub.

    Parameters
    ----------
    search:
        Hugging Face search phrase. Defaults to ``"ocr"``.
    use_case:
        Optional context such as ``"invoice"``, ``"table"``, or
        ``"handwriting"``. It refines the search and appears in output.
    limit:
        Maximum number of candidates to return (1-50).
    compatible_only:
        Return only candidates estimated to fit detected CUDA VRAM.
    include_unknown:
        Include repositories with no published or discoverable parameter count.
        They cannot receive a reliable VRAM compatibility decision and are
        hidden by default.
    refresh:
        Ignore the short-lived local cache and query Hugging Face again.
    use_cache:
        Enable the 15-minute local response cache. The CLI enables this for
        fast repeat searches; library callers stay deterministic by default.
    profile:
        Optional hardware snapshot, mainly useful for tests and integrations.

    Raises
    ------
    ImportError
        If the lightweight ``catalog`` extra is not installed.
    RuntimeError
        If the Hugging Face request fails.
    """
    limit = max(1, min(int(limit), 50))
    query = " ".join(part for part in (search.strip(), (use_case or "").strip()) if part)
    cache_enabled = use_cache
    active_profile = profile or inspect_hardware()
    key = _cache_key(
        search, use_case, limit, compatible_only, include_unknown, active_profile
    )
    if cache_enabled and not refresh:
        cached_models = _read_cache(key)
        if cached_models is not None:
            return cached_models

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import disable_progress_bars
    except ImportError as exc:
        raise ImportError(
            "Online model discovery requires the catalog extra. Install with: "
            'pip install "textlens-srevarshan[catalog]"'
        ) from exc

    try:
        # Keep the first live request bounded.  Follow-up calls are served
        # from the local cache, while --refresh intentionally re-queries Hub.
        api = HfApi()
        raw_models: Iterable[Any] = api.list_models(
            search=query,
            sort="downloads",
            limit=limit,
            full=True,
        )
        raw_models = list(raw_models)
        # Safetensors inspection is read-only, but the Hub client otherwise
        # emits progress bars and a Windows symlink warning into the CLI.
        # TextLens keeps this advisor output focused on the recommendation.
        with warnings.catch_warnings(), disable_progress_bars():
            warnings.filterwarnings(
                "ignore",
                message=".*symlinks by default.*",
                category=UserWarning,
            )
            published_counts = _published_parameter_counts(api, raw_models)
        candidates: List[DiscoveredModel] = []
        for model in raw_models:
            repo_id = str(getattr(model, "modelId", "unknown"))
            official = _official_metadata(repo_id)
            params_b = (
                _parse_parameter_label(official.parameters)
                if official is not None
                else _parameter_count_billion(model) or published_counts.get(repo_id)
            )
            # Official registry entries use their tested minimum VRAM; all
            # third-party candidates use a deliberately conservative estimate.
            estimated_vram = (
                official.min_vram_gb
                if official is not None
                else _estimate_vram_gb(params_b)
            )
            if params_b is None and not include_unknown:
                continue
            compatibility = _compatibility(active_profile, estimated_vram)
            # An explicitly requested unknown-metadata model remains visible
            # for research, but TextLens never claims it fits the GPU.
            if compatible_only and compatibility != "Compatible" and not (
                include_unknown and compatibility == "VRAM not published"
            ):
                continue
            candidates.append(
                DiscoveredModel(
                    repo_id=repo_id,
                    pipeline_tag=getattr(model, "pipeline_tag", None),
                    tags=[str(tag) for tag in (getattr(model, "tags", None) or [])],
                    downloads=int(getattr(model, "downloads", 0) or 0),
                    likes=int(getattr(model, "likes", 0) or 0),
                    parameter_count_b=params_b,
                    estimated_vram_gb=estimated_vram,
                    compatibility=compatibility,
                    use_case_signals=(
                        ([use_case.title()] if use_case else []) + official.use_cases
                        if official is not None
                        else _use_case_signals(model, use_case)
                    ),
                )
            )
            if len(candidates) >= limit:
                break
        if cache_enabled:
            _write_cache(key, candidates)
        return candidates
    except Exception as exc:
        raise RuntimeError(f"Could not query Hugging Face model catalog: {exc}") from exc


def print_discovered_models(
    models: List[DiscoveredModel],
    profile: HardwareProfile,
) -> None:
    """Render live candidates in Rich when available, otherwise plain text."""
    def compact_signals(item: DiscoveredModel) -> str:
        """Keep terminal tables readable while retaining the primary matches."""
        if len(item.use_case_signals) <= 4:
            return ", ".join(item.use_case_signals)
        return ", ".join(item.use_case_signals[:4]) + ", ..."

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        table = Table(
            title="[bold cyan]TextLens Live Model Finder[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold bright_cyan",
            padding=(0, 1),
            show_lines=False,
        )
        table.add_column("Repository", style="cyan", overflow="fold")
        table.add_column("Parameters", justify="right", style="bold white")
        table.add_column("VRAM guide", justify="right", style="yellow")
        table.add_column("Hardware fit", min_width=15)
        table.add_column("Use-case signals", overflow="fold")
        for item in models:
            params = f"{item.parameter_count_b:g}B" if item.parameter_count_b is not None else "Unknown"
            vram = f"~{item.estimated_vram_gb:g} GB" if item.estimated_vram_gb is not None else "Unknown"
            fit_style = (
                "bold green" if item.compatibility == "Compatible"
                else "bold yellow" if item.compatibility.startswith("Needs")
                else "dim"
            )
            table.add_row(
                item.repo_id,
                params,
                vram,
                Text(item.compatibility, style=fit_style),
                compact_signals(item),
            )
        console = Console(highlight=False)
        console.print(
            f"[bold]Hardware[/bold]  [cyan]{profile.primary_gpu_name or 'CPU'}[/cyan] "
            f"[dim]({profile.primary_vram_gb:g} GB VRAM)[/dim]"
        )
        console.print(table)
    except ImportError:
        print(f"Detected hardware: {profile.primary_gpu_name or 'CPU'} ({profile.primary_vram_gb:g} GB VRAM)")
        for item in models:
            params = f"{item.parameter_count_b:g}B" if item.parameter_count_b is not None else "Unknown"
            vram = f"~{item.estimated_vram_gb:g} GB" if item.estimated_vram_gb is not None else "Unknown"
            print(f"\n{item.repo_id}\n  Params: {params}; VRAM guide: {vram}; Fit: {item.compatibility}")
            print(f"  Pipeline: {item.pipeline_tag or 'Not published'}; Use cases: {compact_signals(item)}")
