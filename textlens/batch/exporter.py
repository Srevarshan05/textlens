"""
textlens.batch.exporter
───────────────────────
Structured Exporters for BatchOCR results.

Supports exporting batch results into:
- JSON (.json)
- Markdown (.md)
- Tabular CSV (.csv)
- Clean Plain Text (.txt)
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from textlens.batch.types import BatchTask

logger = logging.getLogger("textlens.batch.exporter")


class StructuredExporter:
    """Handles structured exporting of individual tasks and full batch manifests."""

    def __init__(self, output_dir: Union[str, Path], default_format: str = "json") -> None:
        self.output_dir = Path(output_dir)
        self.default_format = default_format.lower().strip()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_task(
        self,
        task: BatchTask,
        model_id: str,
        output_format: Optional[str] = None,
    ) -> Path:
        """Export an individual completed task's OCR result to disk.

        Parameters
        ----------
        task : BatchTask
            Completed batch task.
        model_id : str
            Model ID used for inference.
        output_format : str, optional
            Export format ("json", "markdown", "csv", "txt"). Defaults to configured default format.

        Returns
        -------
        Path
            Path to the saved result file.
        """
        fmt = (output_format or self.default_format).lower().strip()
        stem = task.source_path.stem
        out_filename = f"{stem}_ocr.{'md' if fmt in ('markdown', 'md') else fmt}"
        out_path = self.output_dir / out_filename

        text_content = task.result_text or ""

        if fmt == "json":
            data = {
                "task_id": task.task_id,
                "file_name": task.source_path.name,
                "source_path": str(task.source_path),
                "model_id": model_id,
                "duration_sec": round(task.duration_sec, 3),
                "page_count": task.page_count,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.completed_at or time.time())),
                "text": text_content,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif fmt in ("markdown", "md"):
            header = (
                f"# OCR Output: {task.source_path.name}\n\n"
                f"- **Source**: `{task.source_path}`\n"
                f"- **Model**: `{model_id}`\n"
                f"- **Pages**: {task.page_count}\n"
                f"- **Processing Time**: {task.duration_sec:.2f}s\n\n"
                f"---\n\n"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header + text_content + "\n")

        elif fmt == "csv":
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_name", "source_path", "model_id", "duration_sec", "text"])
                writer.writerow([task.source_path.name, str(task.source_path), model_id, f"{task.duration_sec:.3f}", text_content])

        else:  # plain txt
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text_content + "\n")

        task.output_path = out_path
        return out_path

    def export_summary_manifest(
        self,
        tasks: List[BatchTask],
        model_id: str,
        elapsed_sec: float,
    ) -> Path:
        """Export a consolidated summary manifest JSON of the entire batch job.

        Parameters
        ----------
        tasks : list[BatchTask]
            All tasks in the batch.
        model_id : str
            Active model ID.
        elapsed_sec : float
            Total batch execution duration.

        Returns
        -------
        Path
            Path to the saved batch_manifest.json file.
        """
        manifest_path = self.output_dir / "batch_manifest.json"

        completed = [t for t in tasks if t.status.value == "COMPLETED"]
        failed = [t for t in tasks if t.status.value == "FAILED"]

        manifest_data = {
            "batch_summary": {
                "total_files": len(tasks),
                "completed_files": len(completed),
                "failed_files": len(failed),
                "elapsed_time_sec": round(elapsed_sec, 2),
                "average_speed_fps": round(len(completed) / max(elapsed_sec, 0.001), 2),
                "model_id": model_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "tasks": [t.to_dict() for t in tasks],
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        logger.info("Exported batch manifest to %s", manifest_path)
        return manifest_path
