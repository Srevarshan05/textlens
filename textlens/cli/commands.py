"""Implementations of the ``textlens`` sub-commands (imported lazily)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from textlens.cli.output import console, emit, emit_json, info, parse_pages

# ── shared helpers ───────────────────────────────────────────────────────────


def make_ocr(args: argparse.Namespace, **overrides: Any) -> Any:
    from textlens.core.engine import OCR

    kwargs: Dict[str, Any] = {}
    for name in ("profile", "model", "device", "backend", "endpoint", "dpi", "min_confidence", "task"):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    if getattr(args, "ocr", None):
        kwargs["ocr"] = args.ocr
    if getattr(args, "fallback", None) is not None:
        kwargs["fallback"] = args.fallback
    if getattr(args, "password", None):
        kwargs["password"] = args.password
    if getattr(args, "ocr_images", False):
        kwargs["ocr_images"] = True
    if getattr(args, "no_download", False):
        kwargs["auto_download"] = False
    kwargs.update(overrides)
    return OCR(**kwargs)


def render(result: Any, fmt: str) -> str:
    fmt = fmt.lower()
    if fmt in ("json",):
        return result.to_json()
    if fmt in ("md", "markdown"):
        return result.to_markdown(page_markers=result.page_count > 1)
    if fmt == "html":
        return result.to_html()
    if fmt == "csv":
        return result.to_csv()
    return result.text


def _progress_events(total_hint: Optional[int] = None):  # type: ignore[no-untyped-def]
    """A (callback, close) pair driving a Rich progress bar on stderr."""
    c = console(stderr=True)
    if c is None or not sys.stderr.isatty():
        return None, lambda: None
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    progress = Progress(TextColumn("[cyan]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=c, transient=True)
    state: Dict[str, Any] = {"task": None}

    def on_event(event: str, payload: Dict[str, Any]) -> None:
        if event == "document_start":
            progress.start()
            state["task"] = progress.add_task("pages", total=payload.get("pages"))
        elif event == "page_done" and state["task"] is not None:
            progress.update(state["task"], advance=1, description=f"page {payload.get('page')} · {payload.get('source')}")

    def close() -> None:
        if state["task"] is not None:
            progress.stop()

    return on_event, close


def print_explanations(result: Any) -> None:
    for page in result.pages:
        prov = page.provenance
        head = f"[bold]Page {page.number}[/bold] · {page.classification or '-'} · source=[cyan]{prov.source}[/cyan]"
        if prov.model:
            head += f" · model=[green]{prov.model}[/green]"
        if prov.confidence is not None:
            head += f" · confidence={prov.confidence:.3f}"
        info(head)
        if prov.reasons:
            info(f"  OCR reasons: {', '.join(prov.reasons)}")
        if prov.routing:
            for line in prov.routing.splitlines():
                info(f"  {line}")
        if len(prov.attempts) > 1:
            info("  attempts: " + " → ".join(f"{a.model} ({a.reason})" for a in prov.attempts))
        for w in prov.warnings:
            info(f"  [yellow]! {w}[/yellow]")


# ── ocr / document / read ────────────────────────────────────────────────────


def cmd_ocr(args: argparse.Namespace, default_format: str = "text") -> None:
    fmt = args.format or default_format
    ocr = make_ocr(args)
    sources: List[str] = args.sources
    out_dir = Path(args.output) if (args.output and len(sources) > 1) else None
    for src in sources:
        events, close = _progress_events()
        t0 = time.perf_counter()
        call: Dict[str, Any] = {"pages": parse_pages(args.pages), "cache": False if args.no_cache else None, "events": events}
        prompt = getattr(args, "prompt", None)
        if prompt and prompt != "Text Recognition:":
            call["prompt"] = prompt
        try:
            result = ocr(src, **call)
        finally:
            close()
        if args.explain:
            print_explanations(result)
        text = render(result, fmt)
        if out_dir is not None:
            ext = {"markdown": "md", "text": "txt"}.get(fmt, fmt)
            emit(text, str(out_dir / f"{Path(str(src)).stem}.{ext}"))
        else:
            emit(text, args.output)
        if not args.quiet:
            by_src = result.provenance.routing.get("pages_by_source", {}) if result.provenance.routing else {}
            conf = result.confidence
            info(
                f"[dim]{Path(str(src)).name}: {result.page_count} page(s) {dict(by_src)} · "
                f"confidence {conf:.3f} · {time.perf_counter() - t0:.2f}s · profile {result.provenance.profile}[/dim]"
                if conf is not None
                else f"[dim]{Path(str(src)).name}: {result.page_count} page(s) {dict(by_src)} · {time.perf_counter() - t0:.2f}s[/dim]"
            )


def cmd_document(args: argparse.Namespace) -> None:
    if getattr(args, "task", None) is None:
        args.task = "markdown"
    cmd_ocr(args, default_format="markdown")


# ── inspect ──────────────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    from textlens.core.engine import OCR

    ocr = OCR(profile=args.profile) if args.profile else OCR()
    report = ocr.inspect(args.source, pages=parse_pages(args.pages), sample=args.sample, password=args.password)
    if args.overlay:
        from textlens.evaluation.visualize import render_overlays

        paths = render_overlays(args.source, args.overlay, ocr=make_ocr(args), pages=parse_pages(args.pages))
        info(f"[green]✓[/green] wrote {len(paths)} overlay image(s) to {args.overlay}")
    if args.json:
        emit_json(report.to_dict() if hasattr(report, "to_dict") else report)
        return
    if not hasattr(report, "pages"):
        emit_json(report.to_dict() if hasattr(report, "to_dict") else report)
        return
    c = console()
    if c is None:
        print(report.summary())
        for p in report.pages:
            print(f"{p.number:>4} {p.kind:<16} ocr={'yes' if p.needs_ocr else 'no':<3} {','.join(p.reasons) or '-'}")
        return
    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", title=f"[bold]{Path(str(args.source)).name}[/bold]")
    for col, just in (("Page", "right"), ("Kind", "left"), ("OCR", "center"), ("Reasons", "left"), ("Text chars", "right"),
                      ("Images", "right"), ("Img cover", "right"), ("Tables", "center"), ("Math", "center"), ("ms", "right")):
        table.add_column(col, justify=just)
    for p in report.pages:
        table.add_row(
            str(p.number),
            p.kind,
            "[yellow]yes[/yellow]" if p.needs_ocr else "[green]no[/green]",
            ", ".join(p.reasons) or "-",
            str(p.text_chars),
            str(p.image_objects),
            f"{p.image_coverage:.0%}",
            "✓" if p.likely_table else "",
            "✓" if p.likely_formulas else "",
            f"{p.elapsed_ms:.1f}",
        )
    c.print(table)
    c.print(f"[bold]{report.summary()}[/bold]")
    if report.pages_needing_ocr:
        c.print(f"Pages needing OCR: [yellow]{_ranges(report.pages_needing_ocr)}[/yellow] — everything else is extracted natively.")
    else:
        c.print("[green]No OCR needed — the whole document has a usable text layer.[/green]")


def _ranges(pages: List[int]) -> str:
    out, start, prev = [], None, None
    for p in pages + [None]:  # type: ignore[list-item]
        if start is None:
            start = prev = p
        elif p is not None and p == prev + 1:
            prev = p
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = p
    return ", ".join(out)


# ── extract ──────────────────────────────────────────────────────────────────


def _load_schema(value: str) -> Any:
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    value = value.strip()
    if value.startswith(("{", "[")):
        return json.loads(value)
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_extract(args: argparse.Namespace) -> None:
    ocr = make_ocr(args)
    extraction = ocr.extract(args.source, schema=_load_schema(args.schema), strategy=args.strategy, pages=parse_pages(args.pages))
    if args.details:
        emit_json(extraction.to_dict(), args.output)
    else:
        emit_json(extraction.data, args.output)


# ── batch ────────────────────────────────────────────────────────────────────


def cmd_batch(args: argparse.Namespace) -> None:
    from textlens.batch import BatchOCR

    c = console()
    if c:
        c.print(f"\n[bold cyan]TextLens Batch[/bold cyan] — [dim]{args.source}[/dim]")
        c.print(f"  Model:    [yellow]{args.model or 'auto (routed)'}[/yellow]   Profile: [yellow]{args.profile or 'auto'}[/yellow]")
        c.print(f"  Workers:  [yellow]{args.workers}[/yellow]   Format: [yellow]{args.format}[/yellow]   Output: [yellow]{args.output}[/yellow]")
        if not args.no_dashboard:
            c.print(f"  Dashboard: [link=http://127.0.0.1:{args.port}]http://127.0.0.1:{args.port}[/link]\n")
    batch = BatchOCR(
        model=args.model,
        workers=args.workers,
        output_format=args.format,
        output_dir=args.output,
        retries=args.retries,
        dpi=args.dpi,
        device=args.device,
        enable_dashboard=not args.no_dashboard,
        dashboard_port=args.port,
        recursive=not args.no_recursive,
        profile=args.profile,
        resume=args.resume,
    )
    results = batch.run(args.source)
    completed = sum(1 for t in results if t.status.value == "COMPLETED")
    failed = sum(1 for t in results if t.status.value == "FAILED")
    skipped = getattr(batch, "skipped", 0)
    msg = f"Batch complete: {completed} processed, {failed} failed" + (f", {skipped} skipped (resume)" if skipped else "")
    info(f"[bold green]{msg}[/bold green]  → {args.output}")
    if failed:
        sys.exit(1)


# ── models ───────────────────────────────────────────────────────────────────


def _models_table(specs: List[Any], title: str) -> None:
    from textlens.models import artifacts

    c = console()
    rows = []
    for s in specs:
        installed = artifacts.is_installed(s) if s.runnable_locally else None
        rows.append((s, installed))
    if c is None:
        for s, inst in rows:
            state = "installed" if inst else ("remote" if s.status == "remote" else "catalog" if s.status == "catalog" else "not installed")
            print(f"{s.id:<18} {s.backend:<12} {s.parameters or '-':<22} {s.license:<24} {state}")
        return
    from rich import box
    from rich.table import Table

    t = Table(box=box.ROUNDED, header_style="bold cyan", title=title, expand=False)
    t.add_column("Model", no_wrap=True)
    t.add_column("Backend")
    t.add_column("Tasks")
    t.add_column("Hardware")
    t.add_column("License")
    t.add_column("Status", no_wrap=True)
    for s, inst in rows:
        hw = "CPU/edge" if s.edge else (f"GPU ≥{s.min_vram_gb:g} GB" if s.min_vram_gb else ("CPU/GPU" if s.cpu else "GPU"))
        lic = s.license if s.commercial_use == "yes" else f"[yellow]{s.license}[/yellow]"
        if s.status == "remote":
            status = "[blue]served[/blue]"
        elif s.status == "catalog":
            status = "[dim]catalog[/dim]"
        elif inst:
            status = "[green]✓ installed[/green]"
        else:
            size = f"{s.download_size_gb * 1000:.0f} MB" if (s.download_size_gb or 0) < 1 else f"{s.download_size_gb:g} GB"
            status = f"[dim]{size}[/dim]"
        name = f"[bold]{s.id}[/bold]" + (" [dim](default)[/dim]" if s.is_default else "")
        t.add_row(name, s.backend, ", ".join(sorted(s.tasks)), hw, lic, status)
    c.print(t)
    c.print("[dim]Install: textlens models install <model> · Details: textlens models info <model> · Served models: OCR(model=…, endpoint=…)[/dim]")


def cmd_models(args: argparse.Namespace) -> None:
    from textlens.models import artifacts
    from textlens.models.specs import all_specs, get_spec, search_specs

    action = getattr(args, "models_action", None) or "list"
    if action == "list":
        specs = all_specs()
        if getattr(args, "installed", False):
            specs = [s for s in specs if s.runnable_locally and artifacts.is_installed(s)]
        if getattr(args, "task", None):
            specs = [s for s in specs if args.task in s.tasks]
        if getattr(args, "json", False):
            emit_json([dict(s.to_dict(), installed=artifacts.is_installed(s) if s.runnable_locally else None) for s in specs])
        else:
            _models_table(specs, "TextLens model catalog")
    elif action == "search":
        specs = search_specs(" ".join(args.query))
        if getattr(args, "json", False):
            emit_json([s.to_dict() for s in specs])
        elif specs:
            _models_table(specs, f"Models matching '{' '.join(args.query)}'")
        else:
            info(f"No registered model matches '{' '.join(args.query)}'. Try `textlens discover {' '.join(args.query)}` to search Hugging Face.")
    elif action == "install":
        from textlens.models.manager import ModelManager

        for model_id in args.ids:
            spec = get_spec(model_id)
            if spec.status in ("remote", "catalog"):
                info(f"[yellow]{spec.id} is a {spec.status} model; nothing to download.[/yellow] {spec.limitations[0] if spec.limitations else ''}")
                continue
            if args.force:
                from textlens.models.downloader import ModelDownloader

                ModelDownloader().download(spec.id, force=True)
            else:
                ModelManager.download(spec.id)
            from textlens.backends.loader import install_hint, missing_requirements

            if missing_requirements(spec):
                info(f"[yellow]Note:[/yellow] {spec.id} also needs its runtime: [cyan]{install_hint(spec)}[/cyan]")
    elif action == "remove":
        from textlens.models.manager import ModelManager

        for model_id in args.ids:
            ModelManager.remove(get_spec(model_id).id)
    elif action == "info":
        spec = get_spec(args.id)
        if getattr(args, "json", False):
            emit_json(dict(spec.to_dict(), installed=artifacts.is_installed(spec) if spec.runnable_locally else None))
            return
        _model_card(spec)
    elif action == "verify":
        spec = get_spec(args.id)
        if not spec.artifacts:
            info(f"{spec.id} has no pinned artifacts; integrity is managed by the Hugging Face Hub cache.")
            return
        result = artifacts.verify(spec)
        for name, ok in result.items():
            info(f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} {name}")
        if not all(result.values()):
            info(f"[red]Verification failed.[/red] Reinstall with: textlens models install {spec.id} --force")
            sys.exit(1)
    elif action == "path":
        emit(str(artifacts.model_dir(get_spec(args.id))))


def _model_card(spec: Any) -> None:
    from textlens.backends.loader import install_hint, missing_requirements
    from textlens.models import artifacts

    c = console()
    rows = [
        ("ID", spec.id),
        ("Name", spec.display_name),
        ("Provider", spec.provider),
        ("Family", spec.family),
        ("Backend", spec.backend),
        ("Status", spec.status),
        ("Parameters", spec.parameters or "-"),
        ("Tasks", ", ".join(sorted(spec.tasks))),
        ("Languages", ", ".join(spec.languages)),
        ("Outputs", ", ".join(k for k, v in (("boxes", spec.boxes), ("confidence", spec.confidence), ("markdown", spec.markdown)) if v) or "text"),
        ("CPU / GPU", f"CPU: {'yes' if spec.cpu else 'no'}{' (practical)' if spec.cpu_practical else ''} · CUDA: {'yes' if spec.cuda else 'no'} · edge: {'yes' if spec.edge else 'no'}"),
        ("VRAM", f"min {spec.min_vram_gb:g} GB, recommended {spec.recommended_vram_gb or spec.min_vram_gb:g} GB" if spec.min_vram_gb else "not required"),
        ("RAM", f"≥ {spec.min_ram_gb:g} GB"),
        ("Quantization", ", ".join(spec.quantization) or "-"),
        ("License", spec.license + (f" (commercial use: {spec.commercial_use})" if spec.commercial_use != "yes" else "")),
        ("Source", spec.source_url or spec.hf_repo_id or "-"),
        ("Download", f"~{spec.download_size_gb:g} GB" if spec.download_size_gb else "-"),
        ("Installed", ("yes → " + str(artifacts.model_dir(spec))) if spec.runnable_locally and artifacts.is_installed(spec) else "no"),
        ("Runtime", "ready" if not missing_requirements(spec) else f"missing → {install_hint(spec)}"),
        ("Routing tiers", f"quality {spec.quality_tier}/5 · speed {spec.speed_tier}/5 (TextLens heuristics, not benchmarks)"),
    ]
    if c is None:
        for k, v in rows:
            print(f"{k:<14}: {v}")
        print(spec.description)
        for lim in spec.limitations:
            print(f"  - {lim}")
        return
    from rich.panel import Panel
    from rich.table import Table

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="white")
    for k, v in rows:
        t.add_row(k, str(v))
    c.print(Panel(t, title=f"[bold]{spec.display_name}[/bold]", border_style="cyan"))
    c.print(Panel(spec.description, title="About", border_style="dim"))
    if spec.license_notes:
        c.print(f"[yellow]License note:[/yellow] {spec.license_notes}")
    if spec.limitations:
        c.print("[bold]Known limitations[/bold]")
        for lim in spec.limitations:
            c.print(f"  • {lim}")
    c.print("[dim]Benchmarks: see the model card for official figures; measure on your data with `textlens benchmark`.[/dim]")


# ── doctor / setup ───────────────────────────────────────────────────────────


def cmd_doctor(args: argparse.Namespace) -> None:
    from textlens.models.doctor import HardwareDoctor

    doctor = HardwareDoctor()
    report = doctor.run(deep=getattr(args, "deep", False))
    if getattr(args, "json", False):
        emit_json(report.to_dict())
    else:
        doctor.print_report(report)


_USE_CASES = {
    "1": ("edge", "CPU / edge device (Raspberry Pi, Jetson, laptop CPU)", ["ppocrv6-small"], []),
    "2": ("balanced", "NVIDIA GPU workstation", ["ppocrv6-small", "glm-ocr"], ["gpu"]),
    "3": ("document", "Document AI (PDF → Markdown, tables, formulas)", ["ppocrv6-small", "glm-ocr"], ["gpu", "documents"]),
    "4": ("server", "Server / API deployment", ["ppocrv6-small"], ["server"]),
    "5": ("auto", "Development / everything", ["ppocrv6-small"], ["all", "dev"]),
}


def cmd_setup(args: argparse.Namespace) -> None:
    from textlens.config import write_config_file
    from textlens.models import artifacts
    from textlens.runtime.system import inspect_system

    sysinfo = inspect_system(refresh=True)
    hw = sysinfo.hardware
    gpu = f"{hw.primary_gpu_name} ({hw.primary_vram_gb:g} GB)" if hw.primary_gpu_name else "none"
    info(f"\n[bold cyan]Welcome to TextLens setup[/bold cyan]\nDetected: {sysinfo.platform.device_class} · {hw.cpu_logical_cores} threads · {hw.ram_total_gb:.0f} GB RAM · GPU: {gpu}")
    if sysinfo.is_edge:
        suggested = "1"
    elif hw.gpus and hw.primary_vram_gb >= 4:
        suggested = "2"
    else:
        suggested = "1"
    choice = None
    if args.profile:
        choice = next((k for k, v in _USE_CASES.items() if v[0] == args.profile), None)
        if choice is None:
            from textlens.errors import ConfigurationError

            raise ConfigurationError(f"Unknown profile {args.profile!r}", hint="edge, balanced, document, server or auto")
    elif args.yes or not sys.stdin.isatty():
        choice = suggested
    else:
        info("\nHow will you use TextLens?")
        for k, (_, label, _, _) in _USE_CASES.items():
            info(f"  {k}. {label}{'  [green](recommended)[/green]' if k == suggested else ''}")
        answer = input(f"Select [1-5] (default {suggested}): ").strip() or suggested
        choice = answer if answer in _USE_CASES else suggested
    profile, label, models, extras = _USE_CASES[choice]
    info(f"\n→ Profile [bold green]{profile}[/bold green] ({label})")

    # 1. Python extras: printed, installed only with explicit consent.
    from textlens.backends.loader import missing_requirements

    needed_extras = [e for e in extras if e in ("gpu",) and not sysinfo.runtimes.torch] + [e for e in extras if e not in ("gpu",)]
    if needed_extras:
        cmd = f'pip install "textlens-ocr[{",".join(needed_extras)}]"'
        info(f"Optional packages for this profile: [cyan]{cmd}[/cyan]")
        if "gpu" in needed_extras and hw.gpus:
            from textlens.hardware import get_pytorch_cuda_install_cmd

            info(f"CUDA PyTorch for your driver: [cyan]{get_pytorch_cuda_install_cmd(hw.system_cuda_version)}[/cyan]")
        if args.install_extras:
            import subprocess

            ok = args.yes or input("Run this pip command now? [y/N]: ").strip().lower() in ("y", "yes")
            if ok:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"textlens-ocr[{','.join(needed_extras)}]"])

    # 2. Models: download what can run now.
    from textlens.models.specs import find_spec

    for model_id in models:
        spec = find_spec(model_id)
        if spec is None:
            info(f"[dim]Skipping {model_id}: not in this catalog.[/dim]")
            continue
        if missing_requirements(spec):
            info(f"[dim]Skipping {spec.id}: runtime not installed yet.[/dim]")
            continue
        if artifacts.is_installed(spec):
            info(f"[green]✓[/green] {spec.id} already installed")
            continue
        if args.no_download:
            info(f"[dim]Would download {spec.id} (~{spec.download_size_gb:g} GB).[/dim]")
            continue
        info(f"Downloading {spec.id} (~{spec.download_size_gb:g} GB)…")
        artifacts.install(spec)
        info(f"[green]✓[/green] {spec.id} installed and verified")

    # 3. Persist the choice.
    path = write_config_file({"profile": profile})
    info(f"[green]✓[/green] Saved profile to {path}")
    info("\nTry it:  [bold]textlens ocr your-file.pdf[/bold]   ·   diagnose: [bold]textlens doctor[/bold]\n")


# ── serve / mcp ──────────────────────────────────────────────────────────────


def cmd_serve(args: argparse.Namespace) -> None:
    from textlens.serving.app import serve
    from textlens.serving.settings import ServerSettings

    settings = ServerSettings.from_env()
    overrides: Dict[str, Any] = {}
    for key in ("host", "port", "profile", "model", "device", "endpoint", "max_upload_mb", "workers", "max_queue", "rate_limit"):
        v = getattr(args, key, None)
        if v is not None:
            overrides[key] = v
    if args.api_key:
        overrides["api_keys"] = tuple(args.api_key)
    if args.allow_urls:
        overrides["allow_urls"] = True
    if args.no_persist:
        overrides["no_persist"] = True
    if args.cors:
        overrides["cors_origins"] = tuple(args.cors)
    if args.warmup:
        overrides["warmup"] = True
    serve(settings.replace(**overrides))


def cmd_mcp(args: argparse.Namespace) -> None:
    from textlens.serving.mcp import run_mcp

    run_mcp(transport=args.transport, workspace=args.workspace, host=args.host, port=args.port)


# ── profile / benchmark ──────────────────────────────────────────────────────


def cmd_profile(args: argparse.Namespace) -> None:
    from textlens.evaluation.profiler import print_profile, profile_run

    ocr = make_ocr(args)
    report = profile_run(ocr, args.source, pages=parse_pages(args.pages), repeat=args.repeat)
    if args.json:
        emit_json(report)
    else:
        print_profile(report)


def cmd_benchmark(args: argparse.Namespace) -> None:
    from textlens.evaluation.benchmark import print_benchmark, run_benchmark

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    report = run_benchmark(
        args.dataset,
        models=models,
        profiles=[p.strip() for p in args.profiles.split(",")] if args.profiles else None,
        limit=args.limit,
        device=args.device,
        save_measurements=not args.no_save,
        warmup=not args.no_warmup,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        info(f"[green]✓[/green] report written to {args.output}")
    if args.json:
        emit_json(report)
    else:
        print_benchmark(report)


# ── anpr ─────────────────────────────────────────────────────────────────────


def cmd_anpr(args: argparse.Namespace) -> None:
    from textlens.anpr import ANPR

    anpr = ANPR(profile=args.profile, region=args.region, device=args.device, min_confidence=args.min_confidence)
    out = []
    for src in args.sources:
        result = anpr(src)
        if args.annotate:
            dest = Path(args.annotate)
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{Path(str(src)).stem}_anpr.jpg"
            result.annotate(src).save(path)
            info(f"[green]✓[/green] annotated → {path}")
        out.append(dict(result.to_dict(), source=str(src)))
    if args.json or len(out) > 1:
        emit_json(out if len(out) > 1 else out[0])
    else:
        r = out[0]
        if not r["plates"]:
            info("No plates found.")
        for p in r["plates"]:
            emit(f"{p['plate']}\tconfidence={p['confidence']:.3f}\tvalid={p['valid']}\tbbox={p['bbox']}")


# ── cache ────────────────────────────────────────────────────────────────────


def cmd_cache(args: argparse.Namespace) -> None:
    from textlens.runtime.cache import ResultCache

    cache = ResultCache("disk")
    if args.cache_action == "clear":
        n = cache.clear()
        info(f"[green]✓[/green] removed {n} cached result(s)")
    else:
        from textlens.config import get_settings

        d = get_settings().results_dir
        files = list(d.rglob("*.json.gz")) if d.exists() else []
        size = sum(f.stat().st_size for f in files)
        emit_json({"directory": str(d), "entries": len(files), "bytes": size, "mode": get_settings().result_cache})
