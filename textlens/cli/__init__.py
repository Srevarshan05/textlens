"""
textlens.cli
────────────
The ``textlens`` command-line interface.

    textlens ocr image.png                 OCR an image or PDF (selective OCR)
    textlens document paper.pdf            PDF → structured Markdown
    textlens batch ./documents             folders, with optional live dashboard
    textlens inspect paper.pdf             classify pages, see what needs OCR
    textlens extract invoice.pdf -s f.json schema → JSON
    textlens models list|search|install|info|remove|verify|path
    textlens doctor                        diagnose hardware + runtimes
    textlens setup                         guided configuration
    textlens serve                         REST API server
    textlens benchmark ./dataset           compare models on your data
    textlens profile document.pdf          per-stage timings and memory
    textlens anpr car.jpg                  number-plate recognition

Heavy modules are imported only by the command that needs them, so
``textlens --help`` and ``textlens models`` stay instant.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from textlens import __version__
from textlens.cli.output import error as _print_exc
from textlens.cli.output import setup_logging, setup_streams


def _print_error(msg: str) -> None:
    """Print a styled error to stderr (0.x helper, kept for compatibility)."""
    _print_exc(Exception(msg))


# ---------------------------------------------------------------------------
# Handlers (module-level names are part of the test/compat surface)
# ---------------------------------------------------------------------------


def _cmd_models(args: argparse.Namespace) -> None:
    from textlens.cli.commands import cmd_models

    cmd_models(args)


def _cmd_discover(args: argparse.Namespace) -> None:
    """Handle: textlens discover [options] (live Hugging Face search)."""
    from textlens.models.discovery import discover_models, print_discovered_models
    from textlens.models.hardware import inspect_hardware

    interactive = args.interactive or (not args.no_interactive and sys.stdin.isatty() and not args.model_name and args.search == "ocr")
    include_unknown, compatible_only = args.include_unknown, args.compatible_only
    if interactive:
        try:
            search = input("\nSearch Hugging Face model name [popular OCR/VLM models]: ").strip()
            include_unknown = input("Include models without published parameter metadata? [y/N]: ").strip().lower() in {"y", "yes"}
            compatible_only = input("Show only verified models that fit this GPU? [y/N]: ").strip().lower() in {"y", "yes"}
            if search:
                args.search = search
        except (EOFError, KeyboardInterrupt):
            print("\nDiscovery cancelled.")
            return
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
        print(f"No Hugging Face model names matched '{args.model_name or args.search}'.")
        return
    print_discovered_models(models, profile)
    print("\nVRAM guidance is based on published parameter metadata. Live results are research suggestions, "
          "not automatically supported TextLens backends.")


def _cmd_model_install(args: argparse.Namespace) -> None:
    from textlens.models import ModelManager

    ModelManager.download(args.id)


def _cmd_model_remove(args: argparse.Namespace) -> None:
    from textlens.models import ModelManager

    ModelManager.remove(args.id)


def _cmd_model_info(args: argparse.Namespace) -> None:
    from textlens.models import ModelManager

    ModelManager.info(args.id)


def _cmd_doctor(args: argparse.Namespace) -> None:
    from textlens.cli.commands import cmd_doctor

    cmd_doctor(args)


def _cmd_read(args: argparse.Namespace) -> None:
    from textlens.cli.commands import cmd_ocr

    cmd_ocr(args)


def _cmd_ocr(args: argparse.Namespace) -> None:
    from textlens.cli.commands import cmd_ocr

    cmd_ocr(args)


def _cmd_document(args: argparse.Namespace) -> None:
    from textlens.cli.commands import cmd_document

    cmd_document(args)


def _cmd_batch(args: argparse.Namespace) -> None:
    from textlens.cli.commands import cmd_batch

    cmd_batch(args)


def _cmd_serve(args: argparse.Namespace) -> None:
    from textlens.cli.commands import cmd_serve

    cmd_serve(args)


def _dispatch(name: str) -> "callable":  # type: ignore[valid-type]
    def run(args: argparse.Namespace) -> None:
        from textlens.cli import commands

        getattr(commands, f"cmd_{name}")(args)

    return run


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_EXAMPLES = """\
Quick start:
  textlens ocr invoice.png                      print the text
  textlens ocr report.pdf -f markdown -o out.md selective OCR → Markdown file
  textlens document paper.pdf                   structured Markdown (tables, formulas)
  textlens inspect scan.pdf                     which pages need OCR, and why
  textlens ocr scan.pdf --explain               show routing decisions per page
  textlens models list                          model catalog with install status
  textlens doctor                               hardware + runtime diagnosis
  textlens serve --port 8000                    REST API (pip install "textlens-ocr[server]")

Docs: https://github.com/Srevarshan05/textlens/tree/main/docs
"""


def _add_ocr_options(p: argparse.ArgumentParser, formats: List[str]) -> None:
    g = p.add_argument_group("routing")
    g.add_argument("--profile", "-P", choices=["auto", "edge", "fast", "balanced", "accurate", "document", "server"], help="Routing profile (default: auto, or the one saved by `textlens setup`)")
    g.add_argument("--model", "-m", help="Pin a model (see `textlens models list`)")
    g.add_argument("--device", "-d", help="cpu, cuda, cuda:1, tensorrt, coreml, dml")
    g.add_argument("--backend", choices=["openai"], help="Run --model on a remote OpenAI-compatible server")
    g.add_argument("--endpoint", help="Remote server base URL, e.g. http://localhost:8000/v1")
    g.add_argument("--min-confidence", type=float, help="Accept threshold before falling back to another model")
    fb = g.add_mutually_exclusive_group()
    fb.add_argument("--fallback", dest="fallback", action="store_true", default=None, help="Retry low-confidence pages with the next model")
    fb.add_argument("--no-fallback", dest="fallback", action="store_false", help="Never fall back")
    d = p.add_argument_group("document")
    d.add_argument("--pages", "-p", help="Page selection, e.g. 1-3,7,10-")
    d.add_argument("--ocr", choices=["auto", "force", "off"], help="auto = selective OCR (default), force = OCR every page, off = native text only")
    d.add_argument("--dpi", type=int, help="Render resolution for pages that need OCR")
    d.add_argument("--password", help="Password for encrypted PDFs")
    d.add_argument("--ocr-images", action="store_true", help="Also OCR images embedded in native PDF pages")
    d.add_argument("--task", choices=["text", "markdown", "table", "formula"], help="Recognition task for generative models")
    o = p.add_argument_group("output")
    o.add_argument("--format", "-f", choices=formats, help=f"Output format ({', '.join(formats)})")
    o.add_argument("--output", "-o", help="Write to this file (or directory for several inputs)")
    o.add_argument("--explain", action="store_true", help="Print routing and provenance for each page to stderr")
    o.add_argument("--no-cache", action="store_true", help="Ignore the result cache")
    o.add_argument("--no-download", action="store_true", help="Never download models (fail if missing)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="textlens",
        description=f"TextLens {__version__} — OCR without the OCR complexity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )
    parser.add_argument("-v", "-V", "--version", action="version", version=f"textlens {__version__}")
    parser.add_argument("--verbose", action="count", default=0, help="More logging (repeat for debug)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print results and errors")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    fmts = ["text", "markdown", "json", "html", "csv"]

    p = sub.add_parser("ocr", help="OCR images and PDFs (selective OCR, routed models)", formatter_class=argparse.RawDescriptionHelpFormatter,
                       epilog="Examples:\n  textlens ocr receipt.jpg\n  textlens ocr scan.pdf --pages 1-3 -f json -o scan.json\n  textlens ocr a.png b.pdf -o results/")
    p.add_argument("sources", nargs="+", metavar="SOURCE", help="Image/PDF/DOCX/PPTX path or http(s) URL")
    _add_ocr_options(p, fmts)
    p.set_defaults(handler="_cmd_ocr")

    p = sub.add_parser("read", help=argparse.SUPPRESS)  # 0.x name for `ocr`
    p.add_argument("sources", nargs="+", metavar="SOURCE")
    p.add_argument("--prompt", help=argparse.SUPPRESS)
    _add_ocr_options(p, fmts)
    p.set_defaults(handler="_cmd_read")

    p = sub.add_parser("document", help="Convert a document to structured Markdown / JSON / HTML",
                       epilog="Example:\n  textlens document paper.pdf -o paper.md", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", metavar="SOURCE")
    _add_ocr_options(p, fmts)
    p.set_defaults(handler="_cmd_document")

    p = sub.add_parser("inspect", help="Classify a PDF/image without OCR: which pages need OCR and why")
    p.add_argument("source", metavar="SOURCE")
    p.add_argument("--pages", "-p", help="Only these pages")
    p.add_argument("--sample", type=int, help="Inspect N evenly spaced pages (fast triage of huge PDFs)")
    p.add_argument("--password")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.add_argument("--overlay", metavar="DIR", help="Also run the pipeline and save annotated page images (boxes, sources, order)")
    p.add_argument("--profile", "-P", choices=["auto", "edge", "fast", "balanced", "accurate", "document", "server"])
    p.add_argument("--model", "-m")
    p.add_argument("--device", "-d")
    p.set_defaults(handler=_dispatch("inspect"))

    p = sub.add_parser("extract", help="Extract fields to JSON using a schema", formatter_class=argparse.RawDescriptionHelpFormatter,
                       epilog='Examples:\n  textlens extract invoice.pdf -s "invoice_number,date,total"\n  textlens extract invoice.pdf -s schema.json --details')
    p.add_argument("source", metavar="SOURCE")
    p.add_argument("--schema", "-s", required=True, help="JSON file, inline JSON, or comma-separated field names")
    p.add_argument("--strategy", choices=["auto", "vlm", "heuristic"], default="auto")
    p.add_argument("--details", action="store_true", help="Include page, bbox, confidence and method per field")
    _add_ocr_options(p, ["json"])
    p.set_defaults(handler=_dispatch("extract"))

    p = sub.add_parser("batch", help="Process a folder of documents (live dashboard optional)")
    p.add_argument("source", metavar="DIRECTORY_OR_FILE")
    p.add_argument("--model", "-m", default=None, help="Pin a model (default: routed per page)")
    p.add_argument("--profile", "-P", choices=["auto", "edge", "fast", "balanced", "accurate", "document", "server"])
    p.add_argument("--workers", "-w", type=int, default=2, help="Parallel workers sharing one loaded model (default 2)")
    p.add_argument("--format", "-f", default="json", choices=["json", "markdown", "csv", "txt", "html"])
    p.add_argument("--output", "-o", default="./batch_output")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--no-dashboard", action="store_true", help="Disable the live web dashboard")
    p.add_argument("--port", type=int, default=8765, help="Dashboard port")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip files whose output already exists (checkpoint/resume)")
    p.set_defaults(handler="_cmd_batch")

    # models (2.0) and model (0.x)
    models_p = sub.add_parser("models", help="Model catalog: list, search, install, info, remove, verify, path")
    msub = models_p.add_subparsers(dest="models_action", metavar="<action>")
    lp = msub.add_parser("list", help="List all models with install status")
    lp.add_argument("--installed", action="store_true")
    lp.add_argument("--task", help="Only models supporting a task (text, table, formula, markdown…)")
    lp.add_argument("--json", action="store_true")
    sp = msub.add_parser("search", help="Search the catalog")
    sp.add_argument("query", nargs="+")
    sp.add_argument("--json", action="store_true")
    ip = msub.add_parser("install", help="Download (and verify) model weights")
    ip.add_argument("ids", nargs="+", metavar="MODEL")
    ip.add_argument("--force", action="store_true", help="Re-download even if installed")
    rp = msub.add_parser("remove", help="Delete cached weights")
    rp.add_argument("ids", nargs="+", metavar="MODEL")
    np_ = msub.add_parser("info", help="Model card: capabilities, hardware, license, limitations")
    np_.add_argument("id", metavar="MODEL")
    np_.add_argument("--json", action="store_true")
    vp = msub.add_parser("verify", help="Re-hash installed artifacts against pinned SHA-256")
    vp.add_argument("id", metavar="MODEL")
    pp = msub.add_parser("path", help="Print the model's cache directory")
    pp.add_argument("id", metavar="MODEL")
    models_p.set_defaults(handler="_cmd_models")

    model_parser = sub.add_parser("model", help=argparse.SUPPRESS)
    model_sub = model_parser.add_subparsers(dest="model_action", metavar="<action>")
    for name, helptext in (("install", "Download and cache a model"), ("remove", "Delete a cached model"), ("info", "Show model information")):
        mp = model_sub.add_parser(name, help=helptext)
        mp.add_argument("id", metavar="<model-id>")

    discover_p = sub.add_parser("discover", help="Search live Hugging Face OCR/VLM repositories and rate them for this GPU")
    discover_p.add_argument("model_name", nargs="?")
    discover_p.add_argument("--search", default="ocr")
    discover_p.add_argument("--limit", type=int, default=12)
    discover_p.add_argument("--compatible", "--compatible-only", dest="compatible_only", action="store_true")
    discover_p.add_argument("--include-unknown", action="store_true")
    discover_p.add_argument("--interactive", action="store_true")
    discover_p.add_argument("--no-interactive", action="store_true")
    discover_p.add_argument("--refresh", action="store_true")
    discover_p.set_defaults(handler="_cmd_discover")

    p = sub.add_parser("doctor", help="Diagnose hardware, runtimes and model fit")
    p.add_argument("--json", action="store_true")
    p.add_argument("--deep", action="store_true", help="Also import PyTorch to verify CUDA works (slower)")
    p.set_defaults(handler="_cmd_doctor")

    p = sub.add_parser("setup", help="Guided setup: detect hardware, choose a profile, install models")
    p.add_argument("--profile", choices=["edge", "balanced", "document", "server", "auto"])
    p.add_argument("--yes", "-y", action="store_true", help="Accept recommendations without prompting")
    p.add_argument("--install-extras", action="store_true", help="Offer to pip-install the extras for the chosen profile")
    p.add_argument("--no-download", action="store_true", help="Do not download models")
    p.set_defaults(handler=_dispatch("setup"))

    p = sub.add_parser("serve", help="Run the REST API server")
    p.add_argument("--host", default=None, help="Bind address (default 127.0.0.1; use 0.0.0.0 in containers)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--profile", "-P", choices=["auto", "edge", "fast", "balanced", "accurate", "document", "server"])
    p.add_argument("--model", "-m")
    p.add_argument("--device", "-d")
    p.add_argument("--endpoint", help="Remote inference server for served models")
    p.add_argument("--api-key", action="append", help="Require this API key (repeatable; or TEXTLENS_API_KEYS)")
    p.add_argument("--max-upload-mb", type=int)
    p.add_argument("--workers", type=int, help="Concurrent inference workers (default 1 per GPU)")
    p.add_argument("--max-queue", type=int, help="Queued jobs before returning 503 (backpressure)")
    p.add_argument("--rate-limit", type=float, help="Requests per second per client (token bucket)")
    p.add_argument("--allow-urls", action="store_true", help="Allow http(s) URL inputs (SSRF-guarded)")
    p.add_argument("--no-persist", action="store_true", help="Never write uploads or results to disk")
    p.add_argument("--cors", action="append", help="Allowed CORS origin (repeatable)")
    p.add_argument("--warmup", action="store_true", help="Load the default model before accepting traffic")
    p.set_defaults(handler="_cmd_serve")

    p = sub.add_parser("mcp", help="Run TextLens as an MCP server for AI agents (pip install \"textlens-ocr[mcp]\")")
    p.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    p.add_argument("--workspace", default=".", help="Only files under this directory can be read")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(handler=_dispatch("mcp"))

    p = sub.add_parser("profile", help="Per-stage timings, model load time, peak RAM/VRAM for one input")
    p.add_argument("source", metavar="SOURCE")
    p.add_argument("--repeat", type=int, default=1, help="Runs after the first (warm) run")
    p.add_argument("--json", action="store_true")
    _add_ocr_options(p, ["json"])
    p.set_defaults(handler=_dispatch("profile"))

    p = sub.add_parser("benchmark", help="Compare models on your own labelled dataset (CER, WER, latency, memory)")
    p.add_argument("dataset", help="Folder with images/ + ground_truth/ or a manifest.jsonl")
    p.add_argument("--models", help="Comma-separated model ids (default: every runnable local model)")
    p.add_argument("--profiles", help="Comma-separated profiles to compare instead of models")
    p.add_argument("--limit", type=int, help="Use at most N samples")
    p.add_argument("--device", "-d")
    p.add_argument("--output", "-o", help="Write the full JSON report here")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-save", action="store_true", help="Do not store latency measurements for routing")
    p.add_argument("--no-warmup", action="store_true")
    p.set_defaults(handler=_dispatch("benchmark"))

    p = sub.add_parser("anpr", help="Number-plate recognition (pip install \"textlens-ocr[anpr]\")")
    p.add_argument("sources", nargs="+", metavar="IMAGE")
    p.add_argument("--profile", "-P", default="balanced", choices=["edge", "balanced", "accurate"])
    p.add_argument("--region", default="generic", help="Plate format rules: generic, in, eu, uk, us, … or a YAML/JSON file")
    p.add_argument("--device", "-d")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--annotate", metavar="DIR", help="Save annotated images here")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_dispatch("anpr"))

    p = sub.add_parser("cache", help="Result cache: stats or clear")
    p.add_argument("cache_action", choices=["stats", "clear"])
    p.set_defaults(handler=_dispatch("cache"))

    for alias in ("hardware", "info"):  # 0.x aliases for doctor
        sub.add_parser(alias, help=argparse.SUPPRESS).set_defaults(handler="_cmd_doctor", json=False, deep=False)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point (``textlens`` console script and ``python -m textlens``)."""
    setup_streams()
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.quiet)

    if args.command is None:
        parser.print_help()
        return
    if args.command == "model":  # 0.x sub-commands
        if not getattr(args, "model_action", None):
            parser.parse_args(["model", "--help"])
            return
        handler = globals()[f"_cmd_model_{args.model_action}"]
    elif args.command == "models" and getattr(args, "models_action", None) is None:
        args.models_action = "list"
        handler = globals()["_cmd_models"]
    else:
        handler = args.handler
        if isinstance(handler, str):
            handler = globals()[handler]  # looked up at call time (patchable)
    if not hasattr(args, "quiet"):
        args.quiet = False
    try:
        handler(args)
    except KeyboardInterrupt:
        _print_exc(Exception("Interrupted."))
        sys.exit(130)
    except BrokenPipeError:  # e.g. `textlens ocr x | head`
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top-level reporter
        from textlens.errors import BackendUnavailableError, InputError, TextLensError

        if args.verbose >= 2:
            import traceback

            traceback.print_exc()
        _print_exc(exc)
        if isinstance(exc, (InputError, FileNotFoundError, ValueError)):
            sys.exit(2)
        if isinstance(exc, BackendUnavailableError):
            sys.exit(3)
        sys.exit(1 if isinstance(exc, TextLensError) else 1)


if __name__ == "__main__":
    main()
