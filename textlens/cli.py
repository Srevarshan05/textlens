"""
textlens.cli
────────────
Command-line interface for the TextLens OCR Framework.

Subcommands
-----------
textlens models
    List all officially supported TextLens models.

textlens discover
    Search live OCR/VLM candidates from Hugging Face and rank them against
    local hardware.

textlens model install <id>
    Download and cache a model from HuggingFace.

textlens model remove <id>
    Delete a cached model from disk.

textlens model info <id>
    Show detailed metadata, disk usage, and hardware requirements.

textlens doctor
    Inspect hardware and print deterministic model recommendations.

textlens read <source>
    Run OCR on an image file, PDF, or remote URL.

textlens batch <directory>
    Process an entire folder of documents with parallel workers and
    a live local monitoring dashboard.

textlens serve
    Launch the OCR REST API endpoint.
"""

from __future__ import annotations

import sys
import argparse

from textlens import __version__


def _print_error(msg: str) -> None:
    """Print a styled error to stderr."""
    try:
        from rich.console import Console
        Console(stderr=True).print(f"[bold red]Error:[/bold red] {msg}")
    except ImportError:
        print(f"Error: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_models(_args: argparse.Namespace) -> None:
    """Handle: textlens models"""
    from textlens.models.manager import ModelManager
    ModelManager.models()


def _cmd_discover(args: argparse.Namespace) -> None:
    """Handle: textlens discover [options]."""
    from textlens.models.discovery import discover_models, print_discovered_models
    from textlens.models.hardware import inspect_hardware

    interactive = args.interactive or (
        not args.no_interactive
        and sys.stdin.isatty()
        and not args.model_name
        and args.search == "ocr"
    )
    if interactive:
        try:
            try:
                from rich.prompt import Confirm, Prompt

                search = Prompt.ask(
                    "[bold cyan]Search Hugging Face model name[/bold cyan]",
                    default="",
                ).strip()
                include_unknown = Confirm.ask(
                    "[dim]Include repositories without published parameter metadata?[/dim]",
                    default=False,
                )
                compatible_only = Confirm.ask(
                    "[green]Show only verified models that fit this GPU?[/green]",
                    default=False,
                )
            except ImportError:
                search = input("\nSearch Hugging Face model name [popular OCR/VLM models]: ").strip()
                include_unknown = input(
                    "Include models without published parameter metadata? [y/N]: "
                ).strip().lower() in {"y", "yes"}
                compatible_only = input(
                    "Show only verified models that fit this GPU? [y/N]: "
                ).strip().lower() in {"y", "yes"}
            if search:
                args.search = search
        except (EOFError, KeyboardInterrupt):
            print("\nDiscovery cancelled.")
            return
    else:
        include_unknown = args.include_unknown
        compatible_only = args.compatible_only

    profile = inspect_hardware()
    try:
        models = discover_models(
            search=args.model_name or args.search,
            limit=args.limit,
            compatible_only=compatible_only,
            include_unknown=include_unknown,
            refresh=args.refresh,
            use_cache=True,
            profile=profile,
        )
    except (ImportError, RuntimeError) as exc:
        _print_error(str(exc))
        sys.exit(1)

    if not models:
        query = args.model_name or args.search
        print(f"No Hugging Face model names matched '{query}'.")
        print("Try an exact name such as DeepSeek-OCR, PaddleOCR-VL, or GOT-OCR2_0.")
        return
    print_discovered_models(models, profile)
    print(
        "\nVRAM guidance is based on published parameter metadata. Live results "
        "are research suggestions, not automatically supported TextLens backends."
    )


def _cmd_model_install(args: argparse.Namespace) -> None:
    """Handle: textlens model install <id>"""
    from textlens.models import ModelManager
    from textlens.models.exceptions import UnknownModelError, DownloadError
    try:
        ModelManager.download(args.id)
    except UnknownModelError as exc:
        _print_error(str(exc))
        sys.exit(1)
    except DownloadError as exc:
        _print_error(str(exc))
        sys.exit(1)


def _cmd_model_remove(args: argparse.Namespace) -> None:
    """Handle: textlens model remove <id>"""
    from textlens.models import ModelManager
    from textlens.models.exceptions import UnknownModelError
    try:
        ModelManager.remove(args.id)
    except UnknownModelError as exc:
        _print_error(str(exc))
        sys.exit(1)


def _cmd_model_info(args: argparse.Namespace) -> None:
    """Handle: textlens model info <id>"""
    from textlens.models import ModelManager
    from textlens.models.exceptions import UnknownModelError
    try:
        ModelManager.info(args.id)
    except UnknownModelError as exc:
        _print_error(str(exc))
        sys.exit(1)


def _cmd_doctor(_args: argparse.Namespace) -> None:
    """Handle: textlens doctor"""
    from textlens.models.doctor import HardwareDoctor
    doctor = HardwareDoctor()
    report = doctor.run()
    doctor.print_report(report)


def _cmd_read(args: argparse.Namespace) -> None:
    """Handle: textlens read <source>"""
    # Lazy import to keep CLI startup fast
    from textlens.sdk import TextLens

    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        console = None

    ocr = TextLens(device=args.device)

    if args.source.lower().endswith(".pdf"):
        pages = ocr.read_pdf(args.source, prompt=args.prompt)
        print("\n--- PDF OCR RESULTS ---")
        for page in pages:
            print(f"\n[Page {page['page']} / {page['total_pages']}]")
            print(page["text"])
    else:
        text = ocr.read(args.source, prompt=args.prompt)
        print("\n--- OCR RESULT ---")
        print(text)


def _cmd_serve(args: argparse.Namespace) -> None:
    """Handle: textlens serve"""
    from textlens.server import serve
    serve(host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="textlens",
        description=f"TextLens v{__version__} — Multi-Model OCR Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  textlens models\n"
            "  textlens discover                         # search by model name\n"
            "  textlens discover DeepSeek-OCR --compatible\n"
            "  textlens model install glm-ocr\n"
            "  textlens model info smolvlm\n"
            "  textlens model remove smolvlm\n"
            "  textlens doctor\n"
            "  textlens read invoice.png\n"
            "  textlens serve --port 8000\n"
        ),
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"textlens {__version__}",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ── models ─────────────────────────────────────────────────────────
    sub.add_parser(
        "models",
        help="List all officially supported TextLens models",
    )

    discover_p = sub.add_parser(
        "discover",
        help="Search live Hugging Face OCR/VLM repositories by model name",
    )
    discover_p.add_argument(
        "model_name",
        nargs="?",
        help="Optional model name, e.g. DeepSeek-OCR or PaddleOCR-VL",
    )
    discover_p.add_argument(
        "--search",
        default="ocr",
        help="Model name to search (default: popular OCR/VLM models)",
    )
    discover_p.add_argument("--limit", type=int, default=12, help="Candidates to show (1-50; default: 12)")
    discover_p.add_argument(
        "--compatible", "--compatible-only",
        dest="compatible_only",
        action="store_true",
        help="Only show models whose estimated VRAM fits detected CUDA hardware",
    )
    discover_p.add_argument(
        "--include-unknown",
        action="store_true",
        help="Also show repositories without published parameter metadata",
    )
    discover_p.add_argument(
        "--interactive",
        action="store_true",
        help="Open the interactive model-search prompt",
    )
    discover_p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not open the prompt when no model name is supplied",
    )
    discover_p.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch fresh results instead of using the 15-minute local cache",
    )

    # ── model (sub-sub-commands) ────────────────────────────────────────
    model_parser = sub.add_parser(
        "model",
        help="Manage individual models (install / remove / info)",
    )
    model_sub = model_parser.add_subparsers(dest="model_action", metavar="<action>")

    # model install
    install_p = model_sub.add_parser("install", help="Download and cache a model")
    install_p.add_argument("id", metavar="<model-id>", help="Model ID (e.g. glm-ocr)")

    # model remove
    remove_p = model_sub.add_parser("remove", help="Delete a cached model from disk")
    remove_p.add_argument("id", metavar="<model-id>", help="Model ID to remove")

    # model info
    info_p = model_sub.add_parser("info", help="Show detailed model information")
    info_p.add_argument("id", metavar="<model-id>", help="Model ID to inspect")

    # ── doctor ─────────────────────────────────────────────────────────
    sub.add_parser(
        "doctor",
        help="Inspect hardware and get model recommendations",
    )

    # ── read ───────────────────────────────────────────────────────────
    read_p = sub.add_parser("read", help="Run OCR on an image file, PDF, or URL")
    read_p.add_argument("source", metavar="<source>", help="Image path, PDF path, or URL")
    read_p.add_argument(
        "--prompt", "-p",
        default="Text Recognition:",
        help="Custom prompt instruction (default: 'Text Recognition:')",
    )
    read_p.add_argument(
        "--device", "-d",
        choices=["cuda", "cpu"],
        default=None,
        help="Device override (auto-detected if omitted)",
    )

    # ── batch ────────────────────────────────────────────────────
    batch_p = sub.add_parser(
        "batch",
        help="Batch process an entire folder of documents with parallel workers",
    )
    batch_p.add_argument("source", metavar="<directory-or-file>", help="Input folder or file path")
    batch_p.add_argument("--model", "-m", default="glm-ocr", help="Model ID (default: glm-ocr)")
    batch_p.add_argument("--workers", "-w", type=int, default=4, help="Parallel worker threads (default: 4)")
    batch_p.add_argument("--format", "-f", default="json",
                         choices=["json", "markdown", "csv", "txt"],
                         help="Output format (default: json)")
    batch_p.add_argument("--output", "-o", default="./batch_output", help="Output directory (default: ./batch_output)")
    batch_p.add_argument("--retries", type=int, default=2, help="Retry limit per file (default: 2)")
    batch_p.add_argument("--dpi", type=int, default=200, help="PDF render DPI (default: 200)")
    batch_p.add_argument("--no-dashboard", action="store_true", help="Disable the live monitoring dashboard")
    batch_p.add_argument("--port", type=int, default=8765, help="Dashboard port (default: 8765)")
    batch_p.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories")
    batch_p.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Device override")

    # ── serve ────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Launch the OCR REST API server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host binding (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # Legacy aliases kept for backwards compatibility
    for alias in ("hardware", "info"):
        sub.add_parser(alias, help=argparse.SUPPRESS)

    return parser


def _cmd_batch(args: argparse.Namespace) -> None:
    """Handle: textlens batch <source>"""
    from textlens.batch import BatchOCR
    try:
        from rich.console import Console
        console = Console()
        console.print(f"\n[bold cyan]TextLens BatchOCR[/bold cyan] — [dim]{args.source}[/dim]")
        console.print(f"  Model:    [yellow]{args.model}[/yellow]")
        console.print(f"  Workers:  [yellow]{args.workers}[/yellow]")
        console.print(f"  Format:   [yellow]{args.format}[/yellow]")
        console.print(f"  Output:   [yellow]{args.output}[/yellow]")
        if not args.no_dashboard:
            console.print(f"  Dashboard: [link=http://127.0.0.1:{args.port}]http://127.0.0.1:{args.port}[/link]\n")
    except ImportError:
        print(f"TextLens BatchOCR  source={args.source}  model={args.model}  workers={args.workers}")

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
    )
    results = batch.run(args.source)

    completed = sum(1 for t in results if t.status.value == "COMPLETED")
    failed = sum(1 for t in results if t.status.value == "FAILED")
    try:
        from rich.console import Console
        c = Console()
        c.print(f"\n[bold green]Batch Complete[/bold green]  ✓ {completed} processed  ✕ {failed} failed")
        c.print(f"Results saved to: [cyan]{args.output}[/cyan]\n")
    except ImportError:
        print(f"Batch complete. {completed} processed, {failed} failed. Output: {args.output}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint registered as the ``textlens`` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "models":
        _cmd_models(args)

    elif args.command == "discover":
        _cmd_discover(args)

    elif args.command == "model":
        if not args.model_action:
            parser.parse_args(["model", "--help"])
            return
        if args.model_action == "install":
            _cmd_model_install(args)
        elif args.model_action == "remove":
            _cmd_model_remove(args)
        elif args.model_action == "info":
            _cmd_model_info(args)

    elif args.command == "doctor":
        _cmd_doctor(args)

    elif args.command == "read":
        _cmd_read(args)

    elif args.command == "batch":
        _cmd_batch(args)

    elif args.command == "serve":
        _cmd_serve(args)

    elif args.command in ("hardware", "info"):
        # Legacy aliases share Doctor's fast driver-based inspection instead
        # of importing the older eager-PyTorch hardware module.
        _cmd_doctor(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
