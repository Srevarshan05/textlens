"""Process all supported images/PDFs in a folder and export JSON results."""

from __future__ import annotations

import argparse
from pathlib import Path

from textlens.batch import BatchOCR, TaskStatus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("batch_output"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    batch = BatchOCR(
        model="glm-ocr",
        workers=args.workers,
        output_dir=args.output,
        output_format="json",
        enable_dashboard=args.dashboard,
    )
    tasks = batch.run(args.folder)
    for task in tasks:
        print(f"{task.source_path.name}: {task.status.value}")
        if task.status is TaskStatus.FAILED:
            print("  Error:", task.error)
    print("Manifest:", args.output / "batch_manifest.json")


if __name__ == "__main__":
    main()
