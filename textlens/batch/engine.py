"""
textlens.batch.engine
─────────────────────
BatchOCR — High-performance parallel batch document processing engine.

This module provides the `BatchOCR` class which:
- Accepts an entire folder or explicit file list of PDFs/images.
- Processes files in parallel via a configurable worker thread pool.
- Retries failed tasks with configurable backoff.
- Tracks progress in real-time via `JobMetrics`.
- Optionally launches a live local monitoring dashboard.
- Exports results to JSON, Markdown, CSV, or plain text.
- Supports pause / resume / cancel / retry-all-failed via public API.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from textlens.batch.exporter import StructuredExporter
from textlens.batch.queue import BaseBatchQueue, MemoryBatchQueue
from textlens.batch.types import (
    BatchJobConfig,
    BatchStatus,
    BatchTask,
    JobMetrics,
    TaskStatus,
)

logger = logging.getLogger("textlens.batch.engine")

# Supported file extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_PDF_EXTS = {".pdf"}
_SUPPORTED_EXTS = _IMAGE_EXTS | _PDF_EXTS


class BatchOCR:
    """High-performance batch OCR processor for folders and file lists.

    Examples
    --------
    Basic usage::

        from textlens.batch import BatchOCR

        batch = BatchOCR(
            model="glm-ocr",
            workers=4,
            output_format="json",
            enable_dashboard=True,
        )
        results = batch.run("./documents/")

    Explicit file list::

        results = batch.run([
            "invoice1.pdf",
            "scan001.png",
            "report.pdf",
        ])

    Advanced with callbacks::

        def on_done(task):
            print(f"Done: {task.source_path.name} — {len(task.result_text)} chars")

        batch = BatchOCR(model="smolvlm", workers=2)
        batch.run("./docs/", on_file_complete=on_done)
    """

    def __init__(
        self,
        model: str = "glm-ocr",
        workers: int = 4,
        output_format: str = "json",
        output_dir: Union[str, Path] = "./batch_output",
        retries: int = 2,
        dpi: int = 200,
        device: Optional[str] = None,
        enable_dashboard: bool = True,
        dashboard_port: int = 8765,
        dashboard_host: str = "127.0.0.1",
        recursive: bool = True,
        max_new_tokens: int = 2048,
        queue_backend: Optional[BaseBatchQueue] = None,
    ) -> None:
        self._config = BatchJobConfig(
            input_source=".",
            output_dir=Path(output_dir),
            model_id=model,
            workers=workers,
            output_format=output_format,
            retries=retries,
            dpi=dpi,
            device=device,
            enable_dashboard=enable_dashboard,
            dashboard_port=dashboard_port,
            dashboard_host=dashboard_host,
            recursive=recursive,
            max_new_tokens=max_new_tokens,
        )
        self._queue: BaseBatchQueue = queue_backend or MemoryBatchQueue()
        self._exporter = StructuredExporter(output_dir, output_format)
        self._metrics = JobMetrics(
            model_id=model,
            output_format=output_format,
            target_workers=workers,
        )

        # ── synchronisation primitives ──────────────────────────────────
        self._pause_event = threading.Event()
        self._pause_event.set()          # set = not paused, clear = paused
        self._stop_event = threading.Event()
        self._metrics_lock = threading.RLock()
        self._log_lock = threading.Lock()

        # ── internal state ──────────────────────────────────────────────
        self._start_time: float = 0.0
        self._worker_threads: List[threading.Thread] = []
        self._dashboard_thread: Optional[threading.Thread] = None
        self._dashboard_server: Any = None
        self._log_buffer: List[str] = []
        self._ocr_instance: Any = None           # Lazy loaded per-worker

        # ── callbacks ──────────────────────────────────────────────────
        self._on_file_complete: Optional[Callable[[BatchTask], None]] = None
        self._on_file_failed: Optional[Callable[[BatchTask], None]] = None

        self._log("BatchOCR initialized. model=%s workers=%d format=%s",
                  model, workers, output_format)

    # ── Public Run API ──────────────────────────────────────────────────────

    def run(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        on_file_complete: Optional[Callable[[BatchTask], None]] = None,
        on_file_failed: Optional[Callable[[BatchTask], None]] = None,
    ) -> List[BatchTask]:
        """Start batch processing and block until completed or cancelled.

        Parameters
        ----------
        source : str | Path | list
            - A directory path (str or Path) — scans for all supported files.
            - A list of file paths to process explicitly.
        on_file_complete : callable, optional
            Called with the completed `BatchTask` when a file finishes successfully.
        on_file_failed : callable, optional
            Called with the failed `BatchTask` when a file exhausts all retries.

        Returns
        -------
        list[BatchTask]
            All processed tasks with final status and result metadata.
        """
        self._on_file_complete = on_file_complete
        self._on_file_failed = on_file_failed
        self._config.input_source = source

        # ── 1. Discover files ───────────────────────────────────────────
        files = self._discover_files(source)
        if not files:
            self._log("No supported files found in source: %s", source)
            return []

        self._log("Discovered %d file(s) to process.", len(files))

        # ── 2. Enqueue tasks ────────────────────────────────────────────
        for path in files:
            task = BatchTask(
                task_id=str(uuid.uuid4()),
                source_path=path,
                max_retries=self._config.retries,
            )
            self._queue.enqueue(task)

        with self._metrics_lock:
            self._metrics.total_files = len(files)
            self._metrics.queued_files = len(files)
            self._metrics.status = BatchStatus.RUNNING

        self._start_time = time.time()

        # ── 3. Start dashboard ──────────────────────────────────────────
        if self._config.enable_dashboard:
            self._start_dashboard()

        # ── 4. Start worker pool ────────────────────────────────────────
        self._stop_event.clear()
        self._pause_event.set()
        self._worker_threads.clear()

        for i in range(self._config.workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)

        self._log("Started %d worker thread(s).", self._config.workers)

        # ── 5. Block until all workers done ────────────────────────────
        for t in self._worker_threads:
            t.join()

        # ── 6. Finalise ─────────────────────────────────────────────────
        elapsed = time.time() - self._start_time
        all_tasks = self._queue.get_all_tasks()

        with self._metrics_lock:
            self._metrics.elapsed_time_sec = elapsed
            if self._metrics.status not in (BatchStatus.CANCELLED,):
                self._metrics.status = BatchStatus.COMPLETED

        # Export consolidated manifest
        manifest = self._exporter.export_summary_manifest(all_tasks, self._config.model_id, elapsed)
        self._log("Batch complete. %d processed, %d failed. Manifest: %s",
                  self._metrics.processed_files,
                  self._metrics.failed_files,
                  manifest)

        return all_tasks

    # ── Control API ─────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Pause processing — in-flight tasks will complete, new ones will wait."""
        self._pause_event.clear()
        with self._metrics_lock:
            self._metrics.status = BatchStatus.PAUSED
        self._log("Batch job paused.")

    def resume(self) -> None:
        """Resume processing from a paused state."""
        self._pause_event.set()
        with self._metrics_lock:
            self._metrics.status = BatchStatus.RUNNING
        self._log("Batch job resumed.")

    def cancel(self) -> None:
        """Signal all workers to stop after their current task finishes."""
        self._stop_event.set()
        self._pause_event.set()  # Unblock any paused workers so they can exit
        with self._metrics_lock:
            self._metrics.status = BatchStatus.CANCELLED
        self._log("Batch job cancelled by user.")

    def retry_failed(self) -> int:
        """Re-enqueue all tasks that ultimately failed (max_retries exhausted).

        Returns
        -------
        int
            Number of tasks re-queued for retry.
        """
        all_tasks = self._queue.get_all_tasks()
        count = 0
        for task in all_tasks:
            if task.status == TaskStatus.FAILED:
                task.retries = 0
                task.error = None
                self._queue.enqueue(task)
                count += 1
        if count > 0:
            self._log("Re-queued %d failed task(s) for retry.", count)
        return count

    def reconfigure(
        self,
        workers: Optional[int] = None,
        output_format: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> None:
        """Reconfigure runtime settings without restarting the batch job.

        Parameters
        ----------
        workers : int, optional
            New target worker thread count.
        output_format : str, optional
            New output format ("json", "markdown", "csv", "txt").
        retries : int, optional
            New retry limit for future tasks.
        """
        if workers is not None:
            self._config.workers = workers
            with self._metrics_lock:
                self._metrics.target_workers = workers
                is_running = self._metrics.status == BatchStatus.RUNNING
            # Spawn additional workers only for an active job. Starting
            # workers while idle eagerly creates/downloads an OCR model and
            # leaves threads polling an empty queue.
            if is_running:
                current = sum(1 for t in self._worker_threads if t.is_alive())
                for _ in range(max(0, workers - current)):
                    t = threading.Thread(target=self._worker_loop, daemon=True)
                    t.start()
                    self._worker_threads.append(t)
            self._log("Worker count reconfigured to %d.", workers)

        if output_format is not None:
            self._config.output_format = output_format
            self._exporter.default_format = output_format
            with self._metrics_lock:
                self._metrics.output_format = output_format
            self._log("Output format reconfigured to %s.", output_format)

        if retries is not None:
            self._config.retries = retries
            self._log("Retry limit reconfigured to %d.", retries)

    # ── Metrics & Logs (used by dashboard) ─────────────────────────────────

    def get_metrics(self) -> JobMetrics:
        """Return the current live metrics snapshot."""
        self._refresh_system_metrics()
        with self._metrics_lock:
            return self._metrics

    def get_tasks(self) -> List[BatchTask]:
        """Return all tasks with their current status."""
        return self._queue.get_all_tasks()

    def get_logs(self, last_n: int = 100) -> List[str]:
        """Return the last N log lines."""
        with self._log_lock:
            return self._log_buffer[-last_n:]

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Main loop executed by each worker thread."""
        # Lazily import OCR here to avoid circular imports
        from textlens.ocr import OCR
        ocr = OCR(
            model=self._config.model_id,
            # Backends auto-select CUDA/CPU only when the value is None.
            # Passing the string "auto" falls through to their CPU path.
            device=self._config.device,
        )

        while not self._stop_event.is_set():
            # ── Pause gate ─────────────────────────────────────
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            # ── Dequeue next task ──────────────────────────────
            task = self._queue.dequeue(block=True, timeout=2.0)
            if task is None:
                # Timeout — check if all tasks are done
                all_tasks = self._queue.get_all_tasks()
                pending = [t for t in all_tasks if t.status in (TaskStatus.QUEUED, TaskStatus.PROCESSING, TaskStatus.RETRYING)]
                if not pending:
                    break
                continue

            with self._metrics_lock:
                self._metrics.active_workers += 1
                self._metrics.queued_files = self._queue.size()

            task.started_at = time.time()
            task.status = TaskStatus.PROCESSING

            self._log("[%s] Processing: %s", threading.current_thread().name, task.source_path.name)

            try:
                # ── Run inference ──────────────────────────────
                t0 = time.time()
                kwargs: Dict[str, Any] = {
                    "dpi": self._config.dpi,
                    "max_new_tokens": self._config.max_new_tokens,
                }
                if self._config.prompt:
                    kwargs["prompt"] = self._config.prompt

                result = ocr.read(str(task.source_path), **kwargs)

                task.completed_at = time.time()
                task.duration_sec = task.completed_at - t0
                task.result_text = result
                task.status = TaskStatus.COMPLETED

                # ── Count PDF pages ────────────────────────────
                if task.source_path.suffix.lower() == ".pdf":
                    try:
                        import pypdfium2 as pdfium
                        doc = pdfium.PdfDocument(str(task.source_path))
                        task.page_count = len(doc)
                        doc.close()
                    except Exception:
                        task.page_count = 1
                else:
                    task.page_count = 1

                # ── Export result ──────────────────────────────
                self._exporter.export_task(task, self._config.model_id, self._config.output_format)

                # ── Update metrics ─────────────────────────────
                with self._metrics_lock:
                    self._metrics.processed_files += 1
                    self._metrics.queued_files = self._queue.size()
                    self._update_speed_eta()

                self._queue.task_done(task)
                self._log("[%s] Done: %s (%.2fs)", threading.current_thread().name,
                          task.source_path.name, task.duration_sec)

                if self._on_file_complete:
                    try:
                        self._on_file_complete(task)
                    except Exception:
                        pass

            except Exception as exc:
                task.error = str(exc)
                task.status = TaskStatus.FAILED
                self._log("[%s] ERROR: %s — %s", threading.current_thread().name,
                          task.source_path.name, exc)

                retried = self._queue.requeue_failed(task)
                if not retried:
                    with self._metrics_lock:
                        self._metrics.failed_files += 1
                    if self._on_file_failed:
                        try:
                            self._on_file_failed(task)
                        except Exception:
                            pass
            finally:
                with self._metrics_lock:
                    self._metrics.active_workers = max(0, self._metrics.active_workers - 1)

        self._log("[%s] Worker exiting.", threading.current_thread().name)

    def _update_speed_eta(self) -> None:
        """Recompute speed and ETA — must be called under _metrics_lock."""
        elapsed = time.time() - self._start_time
        processed = self._metrics.processed_files
        remaining = self._metrics.total_files - processed - self._metrics.failed_files

        if elapsed > 0 and processed > 0:
            fps = processed / elapsed
            self._metrics.processing_speed_fps = fps
            self._metrics.eta_sec = remaining / fps if fps > 0 else 0.0
        self._metrics.elapsed_time_sec = elapsed

    def _refresh_system_metrics(self) -> None:
        """Update CPU / RAM / VRAM readings in the metrics snapshot."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            with self._metrics_lock:
                self._metrics.cpu_percent = cpu
                self._metrics.ram_used_gb = mem.used / (1024 ** 3)
                self._metrics.ram_total_gb = mem.total / (1024 ** 3)
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                free, total = torch.cuda.mem_get_info(0)
                with self._metrics_lock:
                    self._metrics.vram_total_gb = total / (1024 ** 3)
                    self._metrics.vram_used_gb = (total - free) / (1024 ** 3)
                    self._metrics.gpu_name = props.name
        except Exception:
            pass

        with self._metrics_lock:
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._metrics.elapsed_time_sec = elapsed

    def _discover_files(self, source: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        """Resolve source(s) into a list of supported file Paths."""
        paths: List[Path] = []
        sources = source if isinstance(source, list) else [source]

        for s in sources:
            p = Path(s)
            if p.is_dir():
                pattern = "**/*" if self._config.recursive else "*"
                for f in p.glob(pattern):
                    if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS:
                        paths.append(f.resolve())
            elif p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS:
                paths.append(p.resolve())
            else:
                logger.warning("Skipping unsupported source: %s", s)

        return sorted(set(paths))

    def _start_dashboard(self) -> None:
        """Launch the live monitoring dashboard in a background daemon thread."""
        try:
            from textlens.batch.dashboard import start_dashboard_server
            self._dashboard_thread = threading.Thread(
                target=start_dashboard_server,
                args=(self,),
                kwargs={
                    "host": self._config.dashboard_host,
                    "port": self._config.dashboard_port,
                },
                daemon=True,
                name="BatchDashboard",
            )
            self._dashboard_thread.start()
            self._log(
                "Dashboard started at http://%s:%d",
                self._config.dashboard_host,
                self._config.dashboard_port,
            )
        except Exception as exc:
            logger.warning("Dashboard failed to start: %s", exc)

    def _log(self, msg: str, *args: Any) -> None:
        """Log a message both to the Python logger and to the in-memory buffer."""
        formatted = msg % args if args else msg
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {formatted}"
        logger.info(formatted)
        with self._log_lock:
            self._log_buffer.append(line)
            if len(self._log_buffer) > 2000:
                self._log_buffer = self._log_buffer[-2000:]
