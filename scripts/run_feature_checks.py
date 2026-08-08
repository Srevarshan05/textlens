"""Run TextLens feature checks without downloading a VLM by default.

Examples
--------
python scripts/run_feature_checks.py
python scripts/run_feature_checks.py --image test-image-ocr.png --model glm-ocr --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]


def run_offline_tests() -> None:
    """Run every committed pytest feature contract without model downloads."""
    # Keep temporary artifacts inside the repository. This also makes the
    # runner work in restricted CI/sandbox environments where the user temp
    # directory is unavailable.
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--basetemp",
        str(ROOT / ".test-tmp"),
        "-p",
        "no:cacheprovider",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def run_live_ocr(image: Path, model: str, device: Optional[str]) -> None:
    """Perform one real OCR request; this may download/load model weights."""
    if not image.is_file():
        raise SystemExit(f"Image not found: {image}")

    from textlens import OCR

    print(f"\nRunning live OCR: model={model}, device={device or 'auto'}")
    ocr = OCR(model=model, device=device)
    text = ocr.read(image)
    if not text.strip():
        raise SystemExit("Live OCR returned empty text.")
    print("Live OCR passed. Preview:\n")
    print(text[:500])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TextLens features.")
    parser.add_argument("--image", type=Path, help="Run an additional real OCR request on this image.")
    parser.add_argument("--model", default="glm-ocr", help="Registered model for --image (default: glm-ocr).")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Optional device for --image.")
    args = parser.parse_args()

    print("Running offline TextLens feature suite...")
    run_offline_tests()
    print("Offline feature suite passed.")
    if args.image:
        run_live_ocr(args.image.resolve(), args.model, args.device)
    else:
        print("Live VLM OCR skipped. Add --image PATH to run it.")


if __name__ == "__main__":
    main()
