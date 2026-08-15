"""Live Hugging Face model discovery with hardware-aware recommendations.

The official TextLens registry remains the source of supported inference
backends. This module discovers *candidate* OCR/VLM repositories so developers
can research models that fit their hardware; discovered repositories are not
automatically treated as TextLens-supported backends.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Iterable, List, Optional

from textlens.models.hardware import HardwareProfile, inspect_hardware
from textlens.models.registry import ModelRegistry


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


def _compatibility(profile: HardwareProfile, required_vram_gb: Optional[float]) -> str:
    """Classify a model candidate against locally detected hardware."""
    if required_vram_gb is None:
        return "Unknown VRAM"
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
    profile:
        Optional hardware snapshot, mainly useful for tests and integrations.

    Raises
    ------
    ImportError
        If the lightweight ``catalog`` extra is not installed.
    RuntimeError
        If the Hugging Face request fails.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Online model discovery requires the catalog extra. Install with: "
            'pip install "textlens-srevarshan[catalog]"'
        ) from exc

    limit = max(1, min(int(limit), 50))
    query = " ".join(part for part in (search.strip(), (use_case or "").strip()) if part)
    active_profile = profile or inspect_hardware()

    try:
        # Ask for more than the final limit because hardware filtering may
        # remove candidates after the Hub ranks them by downloads.
        raw_models: Iterable[Any] = HfApi().list_models(
            search=query,
            sort="downloads",
            limit=limit * 3 if compatible_only else limit,
            full=True,
        )
        candidates: List[DiscoveredModel] = []
        for model in raw_models:
            repo_id = str(getattr(model, "modelId", "unknown"))
            official = _official_metadata(repo_id)
            params_b = (
                _parse_parameter_label(official.parameters)
                if official is not None
                else _parameter_count_billion(model)
            )
            # Official registry entries use their tested minimum VRAM; all
            # third-party candidates use a deliberately conservative estimate.
            estimated_vram = (
                official.min_vram_gb
                if official is not None
                else _estimate_vram_gb(params_b)
            )
            compatibility = _compatibility(active_profile, estimated_vram)
            if compatible_only and compatibility != "Compatible":
                continue
            candidates.append(
                DiscoveredModel(
                    repo_id=repo_id,
                    pipeline_tag=getattr(model, "pipeline_tag", None),
                    tags=list(getattr(model, "tags", None) or []),
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
        from rich.console import Console
        from rich.table import Table

        table = Table(title="TextLens - Live Hugging Face OCR/VLM Candidates")
        table.add_column("Repository", style="cyan", overflow="fold")
        table.add_column("Params", justify="right")
        table.add_column("VRAM guide", justify="right")
        table.add_column("Hardware fit")
        table.add_column("Use-case signals", overflow="fold")
        table.add_column("Downloads", justify="right")
        for item in models:
            params = f"{item.parameter_count_b:g}B" if item.parameter_count_b is not None else "Unknown"
            vram = f"~{item.estimated_vram_gb:g} GB" if item.estimated_vram_gb is not None else "Unknown"
            table.add_row(
                item.repo_id,
                params,
                vram,
                item.compatibility,
                compact_signals(item),
                f"{item.downloads:,}",
            )
        Console().print(
            f"Detected hardware: {profile.primary_gpu_name or 'CPU'} "
            f"({profile.primary_vram_gb:g} GB VRAM)"
        )
        Console().print(table)
    except ImportError:
        print(f"Detected hardware: {profile.primary_gpu_name or 'CPU'} ({profile.primary_vram_gb:g} GB VRAM)")
        for item in models:
            params = f"{item.parameter_count_b:g}B" if item.parameter_count_b is not None else "Unknown"
            vram = f"~{item.estimated_vram_gb:g} GB" if item.estimated_vram_gb is not None else "Unknown"
            print(f"\n{item.repo_id}\n  Params: {params}; VRAM guide: {vram}; Fit: {item.compatibility}")
            print(f"  Use cases: {compact_signals(item)}; Downloads: {item.downloads:,}")
