"""
textlens.models.doctor
───────────────────────
``textlens doctor`` — diagnose hardware, runtimes and model fit.

Recommendations are **derived from model specs** (no per-model special
cases): a model is

* **Excellent**        — hardware meets its recommended VRAM, or it is a
                         CPU-practical model on a CPU-only machine;
* **Supported**        — meets the minimum, or runs on CPU albeit slowly;
* **Not Recommended**  — below the minimum, or GPU-only on a CPU machine.

Runtime readiness (is torch / onnxruntime installed?) is reported
separately from hardware fit, with the exact command that fixes it.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Any, Dict, List, Optional

from textlens.models.hardware import HardwareProfile, inspect_hardware
from textlens.models.metadata import ModelMetadata

logger = logging.getLogger("textlens.models.doctor")


class Recommendation(enum.Enum):
    """Model recommendation tier."""

    EXCELLENT = "Excellent"
    SUPPORTED = "Supported"
    NOT_RECOMMENDED = "Not Recommended"


@dataclasses.dataclass(frozen=True)
class ModelRecommendation:
    model: ModelMetadata
    level: Recommendation
    note: str = ""
    ready: Optional[bool] = None  # runtime + weights available
    ready_note: str = ""


@dataclasses.dataclass(frozen=True)
class DoctorReport:
    profile: HardwareProfile
    recommendations: List[ModelRecommendation]
    system: Any = None  # textlens.runtime.system.SystemInfo
    recommended_profile: Optional[str] = None
    recommended_model: Optional[str] = None
    warnings: List[str] = dataclasses.field(default_factory=list)
    fixes: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware": dataclasses.asdict(self.profile),
            "system": self.system.to_dict() if self.system is not None else None,
            "recommended_profile": self.recommended_profile,
            "recommended_model": self.recommended_model,
            "models": [
                {"id": r.model.id, "level": r.level.value, "note": r.note, "ready": r.ready, "ready_note": r.ready_note}
                for r in self.recommendations
            ],
            "warnings": self.warnings,
            "fixes": self.fixes,
        }


def _rule_for_model(model_id: str, vram_gb: float, cuda_available: bool) -> tuple:
    """Return ``(Recommendation, note)`` for a model on the given hardware."""
    from textlens.models.specs import get_spec

    spec = get_spec(model_id)
    E, S, N = Recommendation.EXCELLENT, Recommendation.SUPPORTED, Recommendation.NOT_RECOMMENDED
    if spec.backend == "onnx":
        return E, ("Runs on any CPU; GPU optional." if not cuda_available else "")
    if not cuda_available:
        if not spec.cpu:
            return N, "Requires a CUDA GPU; CPU mode is not practical for this model."
        if spec.cpu_practical:
            return E, "Runs well on CPU."
        return S, "Runs on CPU, but expect significantly slower performance."
    target = spec.recommended_vram_gb or spec.min_vram_gb
    if vram_gb >= target:
        return E, ""
    if vram_gb >= spec.min_vram_gb:
        return S, f"Below the recommended {target:g} GB VRAM; large pages may be slower."
    if spec.cpu_practical:
        return S, f"Only {vram_gb:g} GB VRAM; run it on CPU (device='cpu')."
    return N, f"Needs at least {spec.min_vram_gb:g} GB VRAM."


def _local_specs():
    from textlens.models.specs import all_specs

    return [s for s in all_specs() if s.runnable_locally]


def _evaluate_recommendations(profile: HardwareProfile) -> List[ModelRecommendation]:
    """One recommendation per locally runnable model (catalog order)."""
    out = []
    for spec in _local_specs():
        level, note = _rule_for_model(spec.id, profile.primary_vram_gb, profile.cuda_available)
        out.append(ModelRecommendation(model=spec.to_metadata(), level=level, note=note))
    return out


def _readiness(spec: Any) -> tuple:
    from textlens.backends.loader import install_hint, missing_requirements
    from textlens.models import artifacts

    missing = missing_requirements(spec)
    if missing:
        return False, f"install runtime: {install_hint(spec)}"
    if not artifacts.is_installed(spec):
        size = f" (~{spec.download_size_gb:g} GB)" if spec.download_size_gb else ""
        return False, f"weights not installed{size}: textlens models install {spec.id}"
    return True, "ready"


class HardwareDoctor:
    """Inspect hardware + runtimes and produce model recommendations."""

    def run(self, deep: bool = False) -> DoctorReport:
        from textlens.core.router import Router
        from textlens.runtime.system import SystemInfo, detect_platform, detect_runtimes, inspect_system

        profile = inspect_hardware()
        try:
            system = inspect_system(refresh=True, deep=deep)
        except Exception as exc:  # never let diagnostics crash the doctor
            logger.debug("system inspection failed: %s", exc)
            plat = detect_platform()
            system = SystemInfo(hardware=profile, platform=plat, runtimes=detect_runtimes(plat.system))
        recs = []
        for spec in _local_specs():
            level, note = _rule_for_model(spec.id, profile.primary_vram_gb, profile.cuda_available)
            ready, ready_note = _readiness(spec)
            recs.append(ModelRecommendation(spec.to_metadata(), level, note, ready, ready_note))
        warnings, fixes = self._diagnose(system)
        rec_profile = rec_model = None
        try:
            router = Router("auto", system=system)
            rec_profile = router.profile.name
            rec_model = router.route().model
        except Exception as exc:
            warnings.append(f"No runnable OCR model yet: {getattr(exc, 'message', exc)}")
        return DoctorReport(profile, recs, system, rec_profile, rec_model, warnings, fixes)

    @staticmethod
    def _diagnose(system: Any) -> tuple:
        from textlens.hardware import get_pytorch_cuda_install_cmd

        warnings: List[str] = []
        fixes: List[str] = []
        rt = system.runtimes
        hw = system.hardware
        if not rt.onnxruntime:
            warnings.append("ONNX Runtime is missing: the default edge/CPU OCR engine cannot run.")
            fixes.append("pip install onnxruntime")
        dists = [d for d in ("onnxruntime", "onnxruntime-gpu") if _dist_installed(d)]
        if len(dists) > 1:
            warnings.append("Both onnxruntime and onnxruntime-gpu are installed; they conflict.")
            fixes.append("pip uninstall -y onnxruntime && pip install --force-reinstall onnxruntime-gpu")
        if hw.gpus and rt.torch and rt.torch_cuda_build is False:
            warnings.append(f"An NVIDIA GPU is present but PyTorch {rt.torch} is a CPU-only build.")
            fixes.append(get_pytorch_cuda_install_cmd(hw.system_cuda_version))
        if hw.gpus and not rt.torch:
            fixes.append('For GPU document VLMs: pip install "textlens-ocr[gpu]"  (then install a CUDA torch wheel: '
                         + get_pytorch_cuda_install_cmd(hw.system_cuda_version) + ")")
        if hw.gpus and rt.onnxruntime and "CUDAExecutionProvider" not in rt.ort_providers:
            fixes.append("Optional GPU acceleration for PP-OCR: pip uninstall -y onnxruntime && pip install onnxruntime-gpu")
        if system.platform.device_class == "jetson" and "CUDAExecutionProvider" not in rt.ort_providers:
            fixes.append("Jetson: install NVIDIA's onnxruntime-gpu wheel for your JetPack (see docs/deployment/jetson.md).")
        if system.platform.machine in ("armv7l", "armv6l"):
            warnings.append("32-bit ARM: ONNX Runtime has no official wheels; use a 64-bit OS (see docs/deployment/raspberry-pi.md).")
        if rt.transformers and rt.transformers.split(".")[0].isdigit() and int(rt.transformers.split(".")[0]) < 5:
            warnings.append(f"transformers {rt.transformers}: GLM-OCR, LightOnOCR and HunyuanOCR need transformers >= 5.")
        return warnings, fixes

    def print_report(self, report: DoctorReport) -> None:  # noqa: C901
        try:
            from rich import box
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError:
            self._print_plain(report)
            return
        from textlens import __version__

        console = Console(highlight=False)
        p = report.profile
        sysinfo = report.system
        console.print()
        console.rule(f"[bold cyan]TextLens System Diagnostics[/bold cyan] [dim]v{__version__}[/dim]", style="cyan")
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("Field", style="dim", min_width=16)
        t.add_column("Value", style="bold white")
        t.add_row("OS", p.os_name)
        if sysinfo is not None:
            t.add_row("Platform", f"{sysinfo.platform.device_class} ({sysinfo.platform.machine})" + (f" · {sysinfo.platform.board}" if sysinfo.platform.board else ""))
        t.add_row("Python", p.python_version)
        t.add_row("CPU", f"{p.cpu_name} ({p.cpu_physical_cores}C/{p.cpu_logical_cores}T)")
        t.add_row("RAM", f"{p.ram_total_gb:.1f} GB" if p.ram_total_gb else "unknown (install psutil)")
        t.add_row("GPU", f"{p.primary_gpu_name} · {p.primary_vram_gb:g} GB VRAM" if p.primary_gpu_name else "[dim]none detected[/dim]")
        t.add_row("CUDA driver", p.system_cuda_version or "[dim]-[/dim]")
        console.print(Panel(t, title="[bold]System[/bold]", border_style="cyan"))

        if sysinfo is not None:
            rt = sysinfo.runtimes
            b = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", padding=(0, 2))
            b.add_column("Runtime")
            b.add_column("Status")

            def row(name: str, version: Optional[str], extra: str = "", optional: bool = True) -> None:
                if version:
                    b.add_row(name, Text(f"✓ {version} {extra}".rstrip(), style="green"))
                else:
                    b.add_row(name, Text("optional" if optional else "✗ missing", style="dim" if optional else "bold red"))

            providers = ", ".join(x.replace("ExecutionProvider", "") for x in rt.ort_providers)
            row("ONNX Runtime", rt.onnxruntime, f"[{providers}]" if providers else "", optional=False)
            torch_extra = ""
            if rt.torch:
                torch_extra = "(CUDA)" if sysinfo.torch_cuda else "(CPU build)"
            row("PyTorch", rt.torch, torch_extra)
            row("Transformers", rt.transformers)
            row("vLLM", rt.vllm)
            row("TensorRT", rt.tensorrt)
            row("Paddle", rt.paddle)
            row("FastAPI (server)", rt.fastapi)
            row("ANPR models", rt.fast_plate_ocr and rt.open_image_models and f"{rt.fast_plate_ocr}")
            console.print(Panel(b, title="[bold]Backends[/bold]", border_style="cyan"))

        m = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", padding=(0, 1))
        m.add_column("Model", min_width=16)
        m.add_column("Hardware fit", min_width=16)
        m.add_column("Ready", min_width=6)
        m.add_column("Notes")
        styles = {Recommendation.EXCELLENT: "bold green", Recommendation.SUPPORTED: "bold yellow", Recommendation.NOT_RECOMMENDED: "bold red"}
        for r in report.recommendations:
            ready = Text("yes", style="green") if r.ready else Text("no", style="dim")
            note = r.note if r.ready else (r.ready_note + (f" · {r.note}" if r.note else ""))
            m.add_row(Text(r.model.id, style="bold white"), Text(r.level.value, style=styles[r.level]), ready, Text(note or "-", style="dim"))
        console.print(Panel(m, title="[bold]Local models[/bold]", border_style="cyan"))

        summary = Text()
        summary.append("Recommended profile: ", style="dim")
        summary.append(f"{report.recommended_profile or 'n/a'}\n", style="bold green")
        summary.append("Default model:       ", style="dim")
        summary.append(f"{report.recommended_model or 'n/a'}", style="bold green")
        console.print(Panel(summary, border_style="green"))
        for w in report.warnings:
            console.print(f"[yellow]! {w}[/yellow]")
        if report.fixes:
            console.print("[bold]Suggested commands:[/bold]")
            for f in report.fixes:
                console.print(f"  [cyan]{f}[/cyan]")
        console.print()

    def _print_plain(self, report: DoctorReport) -> None:
        p = report.profile
        sep = "=" * 60
        print(f"\n{sep}\nTextLens Doctor\n{sep}")
        print(f"OS       : {p.os_name}")
        print(f"Python   : {p.python_version}")
        print(f"CPU      : {p.cpu_name} ({p.cpu_physical_cores}C/{p.cpu_logical_cores}T)")
        print(f"RAM      : {p.ram_total_gb:.1f} GB")
        print(f"GPU      : {p.primary_gpu_name or 'Not Detected'} ({p.primary_vram_gb} GB VRAM)")
        print(f"CUDA     : {p.system_cuda_version or 'Not Detected'}")
        print(sep)
        for r in report.recommendations:
            ready = "ready" if r.ready else (r.ready_note or "not ready")
            print(f"  {r.model.id:<16} {r.level.value:<16} {ready}{('  (' + r.note + ')') if r.note else ''}")
        print(sep)
        print(f"Recommended profile: {report.recommended_profile}   default model: {report.recommended_model}")
        for w in report.warnings:
            print(f"! {w}")
        for f in report.fixes:
            print(f"  $ {f}")
        print()


def _dist_installed(name: str) -> bool:
    import importlib.metadata

    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
