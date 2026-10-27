"""Use an instruction prompt to focus OCR on selected document fields."""

from __future__ import annotations

import argparse
from pathlib import Path

from textlens import OCR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    prompt = (
        "Read this receipt. Return the vendor, invoice number, date, and total. "
        "Use one label per line and do not invent missing values."
    )
    result = OCR(model="glm-ocr").read(args.image, prompt=prompt, max_new_tokens=512)
    print(result)


if __name__ == "__main__":
    main()
