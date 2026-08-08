"""Extract page-by-page text from a local PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from textlens import TextLens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--max-pages", type=int, default=3)
    args = parser.parse_args()

    engine = TextLens()
    pages = engine.read_pdf(args.pdf, max_pages=args.max_pages)
    for page in pages:
        print(f"\n--- Page {page['page']} of {page['total_pages']} ---")
        print(page["text"])


if __name__ == "__main__":
    main()
