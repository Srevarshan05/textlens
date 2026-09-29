"""
textlens.evaluation.profiler
────────────────────────────
``textlens profile <file>``: where does the time and memory go?

Runs the file once cold (model load included) and ``repeat`` times warm,
then reports per-stage timings from page provenance (inspect, native
extraction, render, detect, recognize, generate), model load time, peak RAM
and this process's peak GPU memory.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional

from textlens.evaluation.resources import ResourceSampler


def _stages(result: Any) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for page in result.pages:
        for k, v in page.provenance.timings_ms.items():
            totals[k] = totals.get(k, 0.0) + float(v)
    return {k: round(v, 1) for k, v in totals.items()}


def profile_run(ocr: Any, source: Any, pages: Optional[List[int]] = None, repeat: int = 1) -> Dict[str, Any]:
    from textlens.backends.loader import get_pool
    from textlens.inputs.source import open_source

    src = open_source(source)
    loaded_before = {b.name for b in get_pool().loaded()}
    runs = []
    with ResourceSampler() as sampler:
        for i in range(1 + max(0, repeat)):
            t0 = time.perf_counter()
            result = ocr(src, pages=pages, cache=False)
            runs.append({"total_ms": round((time.perf_counter() - t0) * 1000, 1), "stages_ms": _stages(result), "cold": i == 0})
    load_ms = {b.name: b.load_time_ms for b in get_pool().loaded() if b.name not in loaded_before and b.load_time_ms}
    warm = [r["total_ms"] for r in runs[1:]]
    by_source: Dict[str, int] = {}
    for p in result.pages:
        by_source[p.provenance.source] = by_source.get(p.provenance.source, 0) + 1
    return {
        "source": src.name,
        "pages": result.page_count,
        "pages_by_source": by_source,
        "models": result.provenance.models,
        "profile": result.provenance.profile,
        "cold_total_ms": runs[0]["total_ms"],
        "warm_total_ms": round(statistics.fmean(warm), 1) if warm else None,
        "model_load_ms": load_ms,
        "stages_ms": runs[-1]["stages_ms"],
        "per_page_ms": round((statistics.fmean(warm) if warm else runs[0]["total_ms"]) / max(1, result.page_count), 1),
        "peak_rss_mb": round(sampler.peak_rss_mb, 1) if sampler.peak_rss_mb else None,
        "peak_vram_mb": round(sampler.peak_vram_mb, 1) if sampler.peak_vram_mb else None,
        "runs": runs,
    }


def print_profile(report: Dict[str, Any]) -> None:
    from textlens.cli.output import console

    c = console()
    lines = [
        ("Source", f"{report['source']} · {report['pages']} page(s) {report['pages_by_source']}"),
        ("Profile / models", f"{report['profile']} · {', '.join(report['models']) or 'native only'}"),
        ("Cold run", f"{report['cold_total_ms']:,.0f} ms (includes model loading)"),
        ("Warm run", f"{report['warm_total_ms']:,.0f} ms" if report["warm_total_ms"] else "-"),
        ("Per page (warm)", f"{report['per_page_ms']:,.0f} ms"),
    ]
    for model, ms in report["model_load_ms"].items():
        lines.append((f"Load {model}", f"{ms:,.0f} ms"))
    for stage, ms in sorted(report["stages_ms"].items(), key=lambda kv: -kv[1]):
        lines.append((f"  {stage}", f"{ms:,.1f} ms"))
    lines.append(("Peak RAM", f"{report['peak_rss_mb']:,.0f} MB" if report["peak_rss_mb"] else "unknown"))
    lines.append(("Peak VRAM (this process)", f"{report['peak_vram_mb']:,.0f} MB" if report["peak_vram_mb"] else "-"))
    if c is None:
        for k, v in lines:
            print(f"{k:<26} {v}")
        return
    from rich.table import Table

    t = Table(show_header=False, box=None, padding=(0, 2), title="[bold]TextLens profile[/bold]")
    t.add_column(style="dim")
    t.add_column(style="white")
    for k, v in lines:
        t.add_row(k, v)
    c.print(t)
