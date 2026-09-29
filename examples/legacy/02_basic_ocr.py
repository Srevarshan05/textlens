"""Run general OCR on one image using the recommended registry API."""

from __future__ import annotations

import argparse
from pathlib import Path

from textlens import OCR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="glm-ocr")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()

    ocr = OCR(model=args.model, device=args.device)
    text = ocr.read(args.image, max_new_tokens=1024)
    print(text)


if __name__ == "__main__":
    main()
