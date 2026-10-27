"""Run table, formula, and structured JSON extraction separately."""

from __future__ import annotations

import argparse
from pathlib import Path

from textlens import TextLens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--mode", choices=("table", "formula", "json"), default="json")
    args = parser.parse_args()

    engine = TextLens()
    if args.mode == "table":
        result = engine.extract_table(args.image)
    elif args.mode == "formula":
        result = engine.extract_formula(args.image)
    else:
        result = engine.extract_json(
            args.image,
            schema='{"vendor": "str", "invoice_number": "str", "total": "float"}',
        )
    print(result)


if __name__ == "__main__":
    main()
