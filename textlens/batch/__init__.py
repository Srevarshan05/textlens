"""
textlens.batch
──────────────
High-performance batch OCR processing engine for TextLens.

Quickstart
----------
::

    from textlens.batch import BatchOCR

    batch = BatchOCR(
        model="glm-ocr",
        workers=4,
        output_format="json",
        enable_dashboard=True,   # opens http://localhost:8765
    )
    results = batch.run("./documents/")

    for task in results:
        print(task.source_path.name, "->", task.status)

Interactive controls (pause / resume / cancel) are available through
both the dashboard UI and the Python API::

    batch.pause()
    batch.resume()
    batch.cancel()
    batch.retry_failed()
    batch.reconfigure(workers=8, output_format="markdown")
"""

from textlens.batch.engine import BatchOCR
from textlens.batch.queue import BaseBatchQueue, MemoryBatchQueue
from textlens.batch.types import (
    BatchJobConfig,
    BatchStatus,
    BatchTask,
    JobMetrics,
    TaskStatus,
)
from textlens.batch.exporter import StructuredExporter
from textlens.batch.dashboard import start_dashboard_server

__all__ = [
    "BatchOCR",
    "BatchStatus",
    "BatchTask",
    "TaskStatus",
    "JobMetrics",
    "BatchJobConfig",
    "BaseBatchQueue",
    "MemoryBatchQueue",
    "StructuredExporter",
    "start_dashboard_server",
]
