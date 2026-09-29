"""Demonstrate explicit runtime switching between CPU and CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from textlens import TextLens, is_cuda_available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    engine = TextLens(device="cpu")
    print(engine.read(args.image))

    if is_cuda_available():
        print(engine.switch_device("cuda"))
        print(engine.read(args.image))
    else:
        print("CUDA is unavailable; CPU-only check completed.")


if __name__ == "__main__":
    main()
