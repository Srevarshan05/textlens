"""
textlens.cli
────────────
Command-line interface for the TextLens OCR Framework.

Subcommands
-----------
textlens models
    List all officially supported TextLens models.

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
    from textlens.models import ModelManager
    ModelManager.models()


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

    # ── serve ──────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Launch the OCR REST API server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host binding (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # Legacy aliases kept for backwards compatibility
    for alias in ("hardware", "info"):
        sub.add_parser(alias, help=argparse.SUPPRESS)

    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint registered as the ``textlens`` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "models":
        _cmd_models(args)

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

    elif args.command == "serve":
        _cmd_serve(args)

    elif args.command in ("hardware", "info"):
        # Legacy fallback
        from textlens.hardware import print_hardware_status
        print_hardware_status()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
