"""
textlens.batch.types
────────────────────
Data structures, status enumerations, and task metadata for BatchOCR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BatchStatus(str, Enum):
    """Execution state of a BatchOCR job."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskStatus(str, Enum):
    """Lifecycle state of an individual file in a batch."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"


@dataclass
class BatchTask:
    """Represents a single document task in a batch job."""
    task_id: str
    source_path: Path
    status: TaskStatus = TaskStatus.QUEUED
    retries: int = 0
    max_retries: int = 2
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_sec: float = 0.0
    error: Optional[str] = None
    result_text: Optional[str] = None
    output_path: Optional[Path] = None
    page_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to a JSON-serializable dictionary."""
        return {
            "task_id": self.task_id,
            "source_path": str(self.source_path),
            "file_name": self.source_path.name,
            "status": self.status.value,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_sec": round(self.duration_sec, 3),
            "error": self.error,
            "result_text_preview": (self.result_text[:150] + "...") if self.result_text and len(self.result_text) > 150 else self.result_text,
            "output_path": str(self.output_path) if self.output_path else None,
            "page_count": self.page_count,
        }


@dataclass
class JobMetrics:
    """Real-time observability snapshot for dashboard and monitoring."""
    status: BatchStatus = BatchStatus.IDLE
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    queued_files: int = 0
    active_workers: int = 0
    target_workers: int = 4
    processing_speed_fps: float = 0.0
    elapsed_time_sec: float = 0.0
    eta_sec: float = 0.0
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    gpu_name: Optional[str] = None
    model_id: str = "glm-ocr"
    output_format: str = "json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "queued_files": self.queued_files,
            "active_workers": self.active_workers,
            "target_workers": self.target_workers,
            "processing_speed_fps": round(self.processing_speed_fps, 2),
            "elapsed_time_sec": round(self.elapsed_time_sec, 1),
            "eta_sec": round(self.eta_sec, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "vram_used_gb": round(self.vram_used_gb, 2),
            "vram_total_gb": round(self.vram_total_gb, 2),
            "gpu_name": self.gpu_name,
            "model_id": self.model_id,
            "output_format": self.output_format,
        }


@dataclass
class BatchJobConfig:
    """Configuration options for a BatchOCR instance."""
    input_source: Union[str, Path, List[Union[str, Path]]]
    output_dir: Union[str, Path] = "./batch_output"
    model_id: str = "glm-ocr"
    workers: int = 4
    output_format: str = "json"  # "json", "markdown", "csv", "txt"
    retries: int = 2
    dpi: int = 200
    device: Optional[str] = None
    enable_dashboard: bool = True
    dashboard_port: int = 8765
    dashboard_host: str = "127.0.0.1"
    recursive: bool = True
    prompt: Optional[str] = None
    max_new_tokens: int = 2048
