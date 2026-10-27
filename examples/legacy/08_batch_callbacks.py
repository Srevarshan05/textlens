"""Use callbacks to observe each BatchOCR completion or permanent failure."""

from __future__ import annotations

import argparse
from pathlib import Path

from textlens.batch import BatchOCR, TaskStatus


def on_complete(task) -> None:
    print(f"Completed: {task.source_path.name} ({task.duration_sec:.2f}s)")


def on_failed(task) -> None:
    print(f"Failed: {task.source_path.name}: {task.error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    batch = BatchOCR(workers=1, enable_dashboard=False, output_format="markdown")
    tasks = batch.run(args.folder, on_file_complete=on_complete, on_file_failed=on_failed)
    succeeded = sum(task.status is TaskStatus.COMPLETED for task in tasks)
    failed = sum(task.status is TaskStatus.FAILED for task in tasks)
    print(f"Finished: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
