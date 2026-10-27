"""
textlens.evaluation.benchmark
─────────────────────────────
Compare OCR models (or profiles) on *your* labelled data.

Dataset layouts (auto-detected)
    dataset/images/*.png  +  dataset/ground_truth/*.txt   (matching stems)
    dataset/*.png         +  dataset/*.txt                 (siblings)
    dataset/manifest.jsonl   {"file": "images/a.png", "text": "…", "table": [[…]]}

Every number in the report is a **TextLens measurement** on this machine
and dataset — not an official model benchmark.  The report embeds an
environment fingerprint so results can be reproduced.  Measured latencies
are saved to ``$TEXTLENS_HOME/measurements.json`` and the router prefers
them over its built-in speed tiers.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from textlens.evaluation.metrics import table_cell_accuracy, text_scores
from textlens.evaluation.resources import ResourceSampler

logger = logging.getLogger("textlens.benchmark")

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}


@dataclass
class Sample:
    id: str
    path: Path
    text: Optional[str] = None
    table: Optional[List[List[str]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def load_dataset(root: Any, limit: Optional[int] = None) -> List[Sample]:
    root = Path(root)
    if not root.exists():
        from textlens.errors import InputNotFoundError

        raise InputNotFoundError(f"Dataset not found: {root}")
    samples: List[Sample] = []
    manifest = root / "manifest.jsonl" if root.is_dir() else root
    if manifest.is_file() and manifest.suffix == ".jsonl":
        base = manifest.parent
        for i, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            f = rec.get("file") or rec.get("image") or rec.get("path")
            samples.append(Sample(str(rec.get("id", i)), base / f, rec.get("text"), rec.get("table"), {k: v for k, v in rec.items() if k not in ("text", "table")}))
    else:
        img_dir = root / "images" if (root / "images").is_dir() else root
        gt_dir = root / "ground_truth" if (root / "ground_truth").is_dir() else img_dir
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in _IMAGE_EXT:
                continue
            gt = gt_dir / f"{f.stem}.txt"
            if gt.is_file():
                samples.append(Sample(f.stem, f, gt.read_text(encoding="utf-8")))
    if not samples:
        from textlens.errors import InputError

        raise InputError(f"No labelled samples found in {root}.", hint="Use images/ + ground_truth/ (same file stems) or a manifest.jsonl.")
    return samples[:limit] if limit else samples


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def environment() -> Dict[str, Any]:
    from textlens import __version__
    from textlens.runtime.system import inspect_system

    s = inspect_system()
    return {
        "textlens": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "device_class": s.platform.device_class,
        "cpu": s.hardware.cpu_name,
        "gpu": s.hardware.primary_gpu_name,
        "vram_gb": s.hardware.primary_vram_gb,
        "ram_gb": s.hardware.ram_total_gb,
        "onnxruntime": s.runtimes.onnxruntime,
        "torch": s.runtimes.torch,
        "measured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def evaluate(ocr: Any, samples: List[Sample], label: str, warmup: bool = True) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    if warmup and samples:
        try:
            ocr(samples[0].path, cache=False)
        except Exception as exc:  # reported per sample below
            logger.warning("warm-up failed for %s: %s", label, exc)
    totals = {"char_edits": 0.0, "ref_chars": 0.0, "word_edits": 0.0, "ref_words": 0.0}
    tables: List[Dict[str, float]] = []
    pages = 0
    confidences: List[float] = []
    with ResourceSampler() as sampler:
        t_all = time.perf_counter()
        for s in samples:
            t0 = time.perf_counter()
            try:
                result = ocr(s.path, cache=False)
            except Exception as exc:  # noqa: BLE001 - benchmark records failures
                rows.append({"id": s.id, "error": getattr(exc, "message", str(exc))})
                continue
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            pages += result.page_count
            row: Dict[str, Any] = {"id": s.id, "latency_ms": round(ms, 1), "pages": result.page_count, "confidence": result.confidence}
            if result.confidence is not None:
                confidences.append(result.confidence)
            if s.text is not None:
                sc = text_scores(s.text, result.text)
                for k in totals:
                    totals[k] += sc[k]
                row.update({k: round(sc[k], 4) for k in ("cer", "wer", "ned", "exact")})
            if s.table is not None and result.tables:
                tsc = table_cell_accuracy(s.table, result.tables[0].grid())
                tables.append(tsc)
                row.update(tsc)
            rows.append(row)
        wall = time.perf_counter() - t_all
    ok = [r for r in rows if "error" not in r]
    labelled = [r for r in ok if "cer" in r]
    summary: Dict[str, Any] = {
        "label": label,
        "samples": len(samples),
        "succeeded": len(ok),
        "failed": len(rows) - len(ok),
        "cer": round(totals["char_edits"] / totals["ref_chars"], 4) if totals["ref_chars"] else None,
        "wer": round(totals["word_edits"] / totals["ref_words"], 4) if totals["ref_words"] else None,
        "exact_match": round(sum(r["exact"] for r in labelled) / len(labelled), 4) if labelled else None,
        "latency_ms_mean": round(statistics.fmean(latencies), 1) if latencies else None,
        "latency_ms_p50": round(_percentile(latencies, 0.5), 1) if latencies else None,
        "latency_ms_p95": round(_percentile(latencies, 0.95), 1) if latencies else None,
        "pages_per_s": round(pages / wall, 3) if wall > 0 and pages else None,
        "peak_rss_mb": round(sampler.peak_rss_mb, 1) if sampler.peak_rss_mb else None,
        "peak_vram_mb": round(sampler.peak_vram_mb, 1) if sampler.peak_vram_mb else None,
        "mean_confidence": round(statistics.fmean(confidences), 4) if confidences else None,
    }
    if tables:
        summary["table_cell_accuracy"] = round(statistics.fmean(t["cell_accuracy"] for t in tables), 4)
        summary["table_structure_match"] = round(statistics.fmean(t["structure_match"] for t in tables), 4)
    # Is confidence informative? Mean confidence of good vs bad reads.
    good = [r["confidence"] for r in labelled if r.get("confidence") is not None and r["cer"] <= 0.05]
    bad = [r["confidence"] for r in labelled if r.get("confidence") is not None and r["cer"] > 0.05]
    if good and bad:
        summary["confidence_gap"] = round(statistics.fmean(good) - statistics.fmean(bad), 4)
    return {"summary": summary, "samples": rows}


def run_benchmark(
    dataset: Any,
    models: Optional[List[str]] = None,
    profiles: Optional[List[str]] = None,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    save_measurements: bool = True,
    warmup: bool = True,
) -> Dict[str, Any]:
    from textlens.backends.loader import get_pool, missing_requirements
    from textlens.core.engine import OCR
    from textlens.models import artifacts
    from textlens.models.specs import all_specs, get_spec

    samples = load_dataset(dataset, limit)
    runs: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}
    if profiles:
        targets = [("profile", p) for p in profiles]
    else:
        if not models:
            models = [s.id for s in all_specs() if s.runnable_locally and not missing_requirements(s) and artifacts.is_installed(s)]
        targets = [("model", m) for m in models]
    for kind, name in targets:
        try:
            if kind == "model":
                spec = get_spec(name)
                if missing_requirements(spec):
                    skipped[name] = f"runtime missing ({', '.join(missing_requirements(spec))})"
                    continue
                ocr = OCR(model=spec.id, device=device, cache="off", fallback=False)
            else:
                ocr = OCR(profile=name, device=device, cache="off")
        except Exception as exc:  # noqa: BLE001
            skipped[name] = getattr(exc, "message", str(exc))
            continue
        logger.info("benchmarking %s %s on %d samples", kind, name, len(samples))
        run = evaluate(ocr, samples, name, warmup=warmup)
        run["summary"]["kind"] = kind
        runs.append(run)
        if kind == "model":
            get_pool().unload(name)  # free memory before the next model
    report = {
        "dataset": str(dataset),
        "environment": environment(),
        "note": "TextLens measurements on this dataset and machine; not official model benchmarks.",
        "runs": runs,
        "skipped": skipped,
    }
    ranked = [r["summary"] for r in runs if r["summary"]["cer"] is not None]
    if ranked:
        best = min(ranked, key=lambda s: (s["cer"], s["latency_ms_p50"] or 0))
        fastest_ok = min((s for s in ranked if s["cer"] <= best["cer"] + 0.01), key=lambda s: s["latency_ms_p50"] or 0)
        report["recommendation"] = {
            "most_accurate": best["label"],
            "best_tradeoff": fastest_ok["label"],
            "reason": f"lowest CER {best['cer']:.2%}; {fastest_ok['label']} is fastest within 1 point of it",
        }
    if save_measurements:
        _save(runs, report["environment"], str(dataset))
    return report


def _save(runs: List[Dict[str, Any]], env: Dict[str, Any], dataset: str) -> None:
    from textlens.config import get_settings

    settings = get_settings()
    if settings.no_persist:
        return
    path = settings.home / "measurements.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    for run in runs:
        s = run["summary"]
        if s.get("kind") != "model" or not s.get("latency_ms_p50"):
            continue
        data[s["label"]] = {
            "latency_ms_p50": s["latency_ms_p50"],
            "cer": s["cer"],
            "pages_per_s": s["pages_per_s"],
            "dataset": dataset,
            "measured_at": env["measured_at"],
            "gpu": env["gpu"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_benchmark(report: Dict[str, Any]) -> None:
    from textlens.cli.output import console

    c = console()
    runs = [r["summary"] for r in report["runs"]]

    def pct(v: Any) -> str:
        return f"{v:.2%}" if isinstance(v, (int, float)) else "-"

    def num(v: Any, unit: str = "") -> str:
        return f"{v:,.0f}{unit}" if isinstance(v, (int, float)) else "-"

    if c is None:
        for s in runs:
            print(f"{s['label']:<18} CER {pct(s['cer'])} WER {pct(s['wer'])} p50 {num(s['latency_ms_p50'], 'ms')} RAM {num(s['peak_rss_mb'], 'MB')}")
        print(report["note"])
        return
    from rich import box
    from rich.table import Table

    t = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", title=f"Benchmark · {report['dataset']}")
    for col in ("Model/profile", "CER", "WER", "Exact", "p50", "p95", "pages/s", "Peak RAM", "Peak VRAM", "Failed"):
        t.add_column(col, justify="left" if col == "Model/profile" else "right")
    for s in sorted(runs, key=lambda s: (s["cer"] if s["cer"] is not None else 9, s["latency_ms_p50"] or 0)):
        t.add_row(s["label"], pct(s["cer"]), pct(s["wer"]), pct(s["exact_match"]), num(s["latency_ms_p50"], " ms"),
                  num(s["latency_ms_p95"], " ms"), f"{s['pages_per_s']:.2f}" if s["pages_per_s"] else "-",
                  num(s["peak_rss_mb"], " MB"), num(s["peak_vram_mb"], " MB"), str(s["failed"]))
    c.print(t)
    env = report["environment"]
    c.print(f"[dim]{report['note']}  ·  {env['cpu']} · GPU {env['gpu'] or 'none'} · textlens {env['textlens']}[/dim]")
    if report.get("recommendation"):
        r = report["recommendation"]
        c.print(f"[bold green]Recommendation:[/bold green] {r['best_tradeoff']} ({r['reason']})")
    for name, why in report.get("skipped", {}).items():
        c.print(f"[yellow]skipped {name}:[/yellow] {why}")
