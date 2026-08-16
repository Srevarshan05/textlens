"""
textlens.models.doctor
───────────────────────
Smart Hardware Doctor — deterministic model recommendations, zero AI.

The Doctor inspects real hardware (via :mod:`textlens.models.hardware`) and
applies a fixed, documented rule-set to recommend, warn, or reject each
registered TextLens model.

Rules (VRAM-based, deterministic)
----------------------------------

VRAM ≥ 8 GB   → fully recommend all 6 models
VRAM ≥ 6 GB   → fully recommend glm-ocr, florence2, smolvlm, paddleocr
                  warn for lighton-ocr, hunyuan-ocr
VRAM ≥ 4 GB   → fully recommend florence2, smolvlm, paddleocr
                  warn for glm-ocr (large PDFs may be slower)
                  not-recommended for lighton-ocr, hunyuan-ocr
VRAM ≥ 2 GB   → fully recommend smolvlm, paddleocr
                  warn for florence2, glm-ocr
                  not-recommended for lighton-ocr, hunyuan-ocr
VRAM < 2 GB   → fully recommend paddleocr, smolvlm (CPU)
                  not-recommended for everything else
CPU-only      → fully recommend paddleocr, smolvlm
                  warn for glm-ocr, florence2
                  not-recommended for lighton-ocr, hunyuan-ocr

Usage
-----
    from textlens.models.doctor import HardwareDoctor

    doctor = HardwareDoctor()
    report = doctor.run()
    doctor.print_report(report)
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Dict, List

from textlens.models.hardware import HardwareProfile, inspect_hardware
from textlens.models.metadata import ModelMetadata
from textlens.models.registry import ModelRegistry

logger = logging.getLogger("textlens.models.doctor")


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class Recommendation(enum.Enum):
    """Model recommendation tier."""

    EXCELLENT = "Excellent"
    SUPPORTED = "Supported"
    NOT_RECOMMENDED = "Not Recommended"


@dataclasses.dataclass(frozen=True)
class ModelRecommendation:
    """Recommendation for a single model."""

    model: ModelMetadata
    level: Recommendation
    note: str = ""


@dataclasses.dataclass(frozen=True)
class DoctorReport:
    """Full doctor report combining hardware profile and model recommendations."""

    profile: HardwareProfile
    recommendations: List[ModelRecommendation]


# ---------------------------------------------------------------------------
# Deterministic rule-set
# ---------------------------------------------------------------------------

# Each rule is a tuple of:
#   (model_id, min_vram_to_be_excellent, note_if_only_supported, note_if_not_recommended)
# Rule evaluation happens in the context of the *full* VRAM tier logic below.


def _evaluate_recommendations(
    profile: HardwareProfile,
) -> List[ModelRecommendation]:
    """Apply deterministic rules and return a recommendation for every model.

    Parameters
    ----------
    profile : HardwareProfile
        The detected hardware profile.

    Returns
    -------
    list[ModelRecommendation]
        One entry per registered model, in catalog order.
    """
    vram = profile.primary_vram_gb
    cuda = profile.cuda_available
    results: List[ModelRecommendation] = []

    for meta in ModelRegistry.all():
        mid = meta.id
        rec, note = _rule_for_model(mid, vram, cuda)
        results.append(ModelRecommendation(model=meta, level=rec, note=note))

    return results


def _rule_for_model(
    model_id: str, vram_gb: float, cuda_available: bool
) -> tuple[Recommendation, str]:
    """Return (Recommendation, note) for a model given actual hardware.

    All logic is purely deterministic — no randomness, no ML inference.
    """
    E = Recommendation.EXCELLENT
    S = Recommendation.SUPPORTED
    N = Recommendation.NOT_RECOMMENDED

    # SmolVLM — 256M, works everywhere
    if model_id == "smolvlm":
        if cuda_available and vram_gb >= 2.0:
            return E, ""
        if cuda_available and vram_gb > 0.0:
            return S, "Very low VRAM — CPU mode recommended."
        return E, "Excellent on CPU — designed for edge devices."

    # GLM OCR — default, needs 6 GB
    if model_id == "glm-ocr":
        if not cuda_available:
            return S, "Running on CPU — expect significantly slower performance."
        if vram_gb >= 6.0:
            return E, ""
        if vram_gb >= 4.0:
            return S, "Below recommended 6 GB VRAM. Large PDFs may be slower."
        if vram_gb >= 2.0:
            return S, "Low VRAM detected. Use smaller batch sizes."
        return N, "Insufficient VRAM. Use smolvlm instead."

    # LightOnOCR — needs 8 GB
    if model_id == "lighton-ocr":
        if not cuda_available:
            return S, "Runs on CPU — significantly slower for large documents."
        if vram_gb >= 8.0:
            return E, ""
        if vram_gb >= 6.0:
            return S, "Below recommended 8 GB VRAM. May require lower batch size."
        return N, "Requires 8 GB VRAM for reliable performance."

    # HunyuanOCR — needs 8 GB, no CPU warning
    if model_id == "hunyuan-ocr":
        if not cuda_available:
            return N, "Requires GPU. CPU mode is not practical for this model."
        if vram_gb >= 8.0:
            return E, ""
        if vram_gb >= 6.0:
            return S, "Below recommended 8 GB VRAM. Reduce batch size."
        return N, "Requires more GPU memory (8 GB minimum recommended)."

    # Fallback for future models — conservative
    return S, "Hardware compatibility unknown for this model."


# ---------------------------------------------------------------------------
# HardwareDoctor
# ---------------------------------------------------------------------------


class HardwareDoctor:
    """Performs hardware inspection and generates deterministic model recommendations.

    Usage
    -----
    ::

        doctor = HardwareDoctor()
        report = doctor.run()
        doctor.print_report(report)
    """

    def run(self) -> DoctorReport:
        """Inspect hardware and compute model recommendations.

        Returns
        -------
        DoctorReport
            Full hardware profile and per-model recommendations.
        """
        profile = inspect_hardware()
        recommendations = _evaluate_recommendations(profile)
        return DoctorReport(profile=profile, recommendations=recommendations)

    def print_report(self, report: DoctorReport) -> None:  # noqa: C901
        """Print a rich, formatted doctor report to the console.

        Parameters
        ----------
        report : DoctorReport
            A report produced by :meth:`run`.
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich import box

            console = Console(force_terminal=True, highlight=False)
            p = report.profile

            # ── System Overview ─────────────────────────────────────────
            console.print()
            console.rule("[bold cyan]TextLens Doctor[/bold cyan]", style="cyan")
            console.print()

            sys_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            sys_table.add_column("Field", style="dim", min_width=22)
            sys_table.add_column("Value", style="bold white")

            sys_table.add_row("Operating System", p.os_name)
            sys_table.add_row("Python", p.python_version)
            sys_table.add_row("PyTorch", p.torch_version)
            cuda_display = p.cuda_version or p.system_cuda_version or "Not Detected"
            sys_table.add_row("CUDA", cuda_display)

            if p.primary_gpu_name:
                sys_table.add_row("GPU", p.primary_gpu_name)
                sys_table.add_row("VRAM", f"{p.primary_vram_gb} GB")
            else:
                sys_table.add_row("GPU", "[dim]Not Detected[/dim]")

            sys_table.add_row("CPU", p.cpu_name)
            sys_table.add_row(
                "CPU Cores",
                f"{p.cpu_physical_cores} physical / {p.cpu_logical_cores} logical",
            )
            sys_table.add_row("RAM", f"{p.ram_total_gb:.1f} GB")
            sys_table.add_row("Device", p.device_type.upper())

            console.print(
                Panel(sys_table, title="[bold]System Information[/bold]", border_style="cyan")
            )

            # ── Hardware Analysis ────────────────────────────────────────
            console.rule("[bold cyan]Hardware Analysis[/bold cyan]", style="cyan")
            console.print()

            gpu_ok = "[green]YES[/green]" if p.gpus else "[red]NO[/red]"
            cuda_ok = "[green]YES[/green]" if p.cuda_available else "[red]NO[/red]"
            console.print(f"  GPU Detected  : {gpu_ok}")
            console.print(f"  CUDA Ready    : {cuda_ok}")
            console.print()

            # ── Model Recommendations ────────────────────────────────────
            rec_table = Table(
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 2),
            )
            rec_table.add_column("Model", min_width=18)
            rec_table.add_column("Verdict", min_width=18)
            rec_table.add_column("Note", min_width=35)

            ICONS = {
                Recommendation.EXCELLENT: "[OK]",
                Recommendation.SUPPORTED: "[WARN]",
                Recommendation.NOT_RECOMMENDED: "[FAIL]",
            }
            STYLES = {
                Recommendation.EXCELLENT: "bold green",
                Recommendation.SUPPORTED: "bold yellow",
                Recommendation.NOT_RECOMMENDED: "bold red",
            }

            for item in report.recommendations:
                icon = ICONS[item.level]
                style = STYLES[item.level]
                rec_table.add_row(
                    Text(item.model.display_name, style="bold white"),
                    Text(f"{icon} {item.level.value}", style=style),
                    Text(item.note or "-", style="dim"),
                )

            console.print(
                Panel(
                    rec_table,
                    title="[bold]Recommended Models[/bold]",
                    border_style="cyan",
                )
            )

            # ── CPU-only notice ──────────────────────────────────────────
            if not p.cuda_available:
                console.print()
                console.print(
                    Panel(
                        Text(
                            "GPU not detected.\n"
                            "TextLens will run on CPU.\n"
                            "Large OCR tasks may be significantly slower.\n\n"
                            "Best options for CPU:\n"
                            "  [OK] SmolVLM     — Excellent\n"
                            "  [WARN] GLM OCR   — Works, expect slower performance.",
                            style="yellow",
                        ),
                        title="[bold yellow]CPU Mode Notice[/bold yellow]",
                        border_style="yellow",
                    )
                )
            console.print()

        except ImportError:
            # Graceful plain-text fallback when rich is not installed
            self._print_plain(report)

    def _print_plain(self, report: DoctorReport) -> None:
        """Plain-text fallback for environments without rich installed."""
        p = report.profile
        sep = "=" * 50
        print(f"\n{sep}")
        print("TextLens Doctor")
        print(sep)
        print(f"OS           : {p.os_name}")
        print(f"Python       : {p.python_version}")
        print(f"PyTorch      : {p.torch_version}")
        print(f"CUDA         : {p.cuda_version or p.system_cuda_version or 'Not Detected'}")
        print(f"GPU          : {p.primary_gpu_name or 'Not Detected'}")
        print(f"VRAM         : {p.primary_vram_gb} GB")
        print(f"CPU          : {p.cpu_name}")
        print(f"CPU Cores    : {p.cpu_physical_cores}P / {p.cpu_logical_cores}L")
        print(f"RAM          : {p.ram_total_gb:.1f} GB")
        print(f"Device       : {p.device_type.upper()}")
        print(sep)
        print("Model Recommendations")
        print(sep)
        ICONS = {
            Recommendation.EXCELLENT: "[OK]",
            Recommendation.SUPPORTED: "[WARN]",
            Recommendation.NOT_RECOMMENDED: "[FAIL]",
        }
        for item in report.recommendations:
            icon = ICONS[item.level]
            note = f"  ({item.note})" if item.note else ""
            print(f"  {icon} {item.model.display_name:<20} {item.level.value}{note}")
        print(sep)
        if not p.cuda_available:
            print(
                "\nGPU not detected. TextLens will run on CPU.\n"
                "Large OCR tasks may be significantly slower.\n"
                "Recommended: SmolVLM"
            )
        print()
