"""
textlens.cli
────────────
Command-line interface for TextLens OCR Framework.
"""

from __future__ import annotations

import sys
import argparse

from textlens import __version__
from textlens.hardware import print_hardware_status, detect_system_cuda
from textlens.sdk import TextLens
from textlens.server import serve


def main() -> None:
    """CLI entrypoint for textlens command."""
    parser = argparse.ArgumentParser(
        prog="textlens",
        description=f"TextLens v{__version__} - Fast GLM-OCR Python Framework & REST Endpoint"
    )
    parser.add_argument("-v", "--version", action="version", version=f"textlens {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: hardware / info / doctor
    subparsers.add_parser("hardware", help="Display GPU (CUDA) & CPU hardware status")
    subparsers.add_parser("info", help="Display GPU (CUDA) & CPU hardware status")
    subparsers.add_parser("doctor", help="Run system CUDA GPU diagnostic and get tailored PyTorch install command")

    # Command: read
    read_parser = subparsers.add_parser("read", help="Run OCR on an image file, PDF, or remote URL")
    read_parser.add_argument("source", type=str, help="Image file path, PDF path, or remote URL")
    read_parser.add_argument(
        "--prompt", "-p", type=str, default="Text Recognition:", help="Custom prompt instruction"
    )
    read_parser.add_argument(
        "--device", "-d", type=str, choices=["cuda", "cpu"], default=None, help="Target device override"
    )

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch the OCR REST API server endpoint")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP binding (defaults to 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command in ("hardware", "info", "doctor"):
        print_hardware_status()
    elif args.command == "read":
        print_hardware_status()
        ocr = TextLens(device=args.device)
        if args.source.lower().endswith(".pdf"):
            pages = ocr.read_pdf(args.source, prompt=args.prompt)
            print("\n--- PDF OCR RESULTS ---")
            for page in pages:
                print(f"\n[Page {page['page']} / {page['total_pages']}]")
                print(page['text'])
        else:
            text = ocr.read(args.source, prompt=args.prompt)
            print("\n--- OCR RESULT ---")
            print(text)
    elif args.command == "serve":
        serve(host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
