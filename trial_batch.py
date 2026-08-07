"""
trial_batch.py
──────────────
Quick demo of TextLens BatchOCR.

Usage:
    python trial_batch.py

Place some PDF/image files in a folder (or use the current directory)
and this script will process them in parallel with the live dashboard.
"""

from textlens.batch import BatchOCR
from pathlib import Path


def on_done(task):
    print(f"  [OK] {task.source_path.name} — {task.duration_sec:.2f}s — {len(task.result_text or '')} chars")


def on_fail(task):
    print(f"  [FAIL] {task.source_path.name} — {task.error}")


if __name__ == "__main__":
    # ── 1. Point at a folder containing PDFs and images
    source_dir = "."  # Change to your target directory

    print("TextLens BatchOCR Demo")
    print("=" * 48)
    print(f"Source: {Path(source_dir).resolve()}")
    print("Dashboard: http://127.0.0.1:8765")
    print("=" * 48)

    batch = BatchOCR(
        model="glm-ocr",
        workers=2,                    # Parallel worker threads
        output_format="json",         # json | markdown | csv | txt
        output_dir="./batch_output",  # Results directory
        retries=2,                    # Retry limit per file
        dpi=200,                      # PDF rendering DPI
        enable_dashboard=True,        # http://localhost:8765
        dashboard_port=8765,
        recursive=True,               # Scan subdirectories
    )

    results = batch.run(
        source=source_dir,
        on_file_complete=on_done,
        on_file_failed=on_fail,
    )

    # ── Print summary
    completed = [t for t in results if t.status.value == "COMPLETED"]
    failed    = [t for t in results if t.status.value == "FAILED"]

    print(f"\nBatch Complete!")
    print(f"  Processed : {len(completed)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Output    : ./batch_output/")
    print(f"  Manifest  : ./batch_output/batch_manifest.json")
