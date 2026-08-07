"""
tests/test_batch.py
───────────────────
Unit tests for the textlens.batch module.

Tests cover:
- MemoryBatchQueue: enqueue, dequeue, requeue_failed, task_done
- StructuredExporter: JSON, Markdown, CSV, TXT exports + manifest
- BatchOCR file discovery and metrics
- Pause / resume / cancel / reconfigure
- Dashboard module importability
"""

from __future__ import annotations

import json
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from textlens.batch.types import (
    BatchStatus,
    BatchTask,
    BatchJobConfig,
    JobMetrics,
    TaskStatus,
)
from textlens.batch.queue import MemoryBatchQueue
from textlens.batch.exporter import StructuredExporter


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_task(path: str = "test.pdf", task_id: str = "task-001") -> BatchTask:
    return BatchTask(task_id=task_id, source_path=Path(path), max_retries=2)


# ── MemoryBatchQueue Tests ───────────────────────────────────────────────────

class TestMemoryBatchQueue:
    def test_enqueue_increases_size(self):
        q = MemoryBatchQueue()
        assert q.size() == 0
        q.enqueue(_make_task())
        assert q.size() == 1

    def test_dequeue_returns_task(self):
        q = MemoryBatchQueue()
        task = _make_task()
        q.enqueue(task)
        got = q.dequeue(block=False)
        assert got is not None
        assert got.task_id == task.task_id

    def test_dequeue_empty_returns_none(self):
        q = MemoryBatchQueue()
        result = q.dequeue(block=True, timeout=0.1)
        assert result is None

    def test_dequeue_sets_processing_status(self):
        q = MemoryBatchQueue()
        q.enqueue(_make_task())
        task = q.dequeue(block=False)
        assert task.status == TaskStatus.PROCESSING

    def test_requeue_failed_increments_retries(self):
        q = MemoryBatchQueue()
        task = _make_task()
        q.enqueue(task)
        _ = q.dequeue(block=False)
        task.error = "Something broke"
        retried = q.requeue_failed(task)
        assert retried is True
        assert task.retries == 1
        assert task.status == TaskStatus.RETRYING

    def test_requeue_failed_exhausts_to_failed(self):
        q = MemoryBatchQueue()
        task = _make_task()
        task.max_retries = 0
        q.enqueue(task)
        _ = q.dequeue(block=False)
        task.error = "Too many errors"
        retried = q.requeue_failed(task)
        assert retried is False
        assert task.status == TaskStatus.FAILED

    def test_get_all_tasks_returns_snapshot(self):
        q = MemoryBatchQueue()
        q.enqueue(_make_task("a.pdf", "t1"))
        q.enqueue(_make_task("b.png", "t2"))
        all_tasks = q.get_all_tasks()
        assert len(all_tasks) == 2
        ids = {t.task_id for t in all_tasks}
        assert "t1" in ids and "t2" in ids

    def test_get_task_by_id(self):
        q = MemoryBatchQueue()
        task = _make_task("c.jpg", "find-me")
        q.enqueue(task)
        found = q.get_task("find-me")
        assert found is not None
        assert found.task_id == "find-me"

    def test_clear_empties_queue(self):
        q = MemoryBatchQueue()
        q.enqueue(_make_task("x.pdf", "x1"))
        q.enqueue(_make_task("y.pdf", "y1"))
        q.clear()
        assert q.size() == 0
        assert q.get_all_tasks() == []

    def test_thread_safe_concurrent_enqueue(self):
        q = MemoryBatchQueue()
        def enq(n):
            for i in range(50):
                q.enqueue(BatchTask(task_id=f"{n}-{i}", source_path=Path(f"f{n}-{i}.png")))

        threads = [threading.Thread(target=enq, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(q.get_all_tasks()) == 200


# ── StructuredExporter Tests ─────────────────────────────────────────────────

class TestStructuredExporter:
    def _make_completed_task(self, tmpdir: Path, fname: str = "invoice.pdf") -> BatchTask:
        task = BatchTask(
            task_id="exp-001",
            source_path=Path(fname),
            status=TaskStatus.COMPLETED,
            result_text="Hello World\nThis is extracted text.",
            duration_sec=1.23,
            page_count=2,
        )
        task.completed_at = time.time()
        return task

    def test_export_json(self, tmp_path):
        exp = StructuredExporter(tmp_path, "json")
        task = self._make_completed_task(tmp_path)
        out = exp.export_task(task, "glm-ocr")
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["text"] == "Hello World\nThis is extracted text."
        assert data["model_id"] == "glm-ocr"
        assert data["duration_sec"] == pytest.approx(1.23, abs=0.01)

    def test_export_markdown(self, tmp_path):
        exp = StructuredExporter(tmp_path, "markdown")
        task = self._make_completed_task(tmp_path)
        out = exp.export_task(task, "lighton-ocr", output_format="markdown")
        assert out.suffix == ".md"
        content = out.read_text(encoding="utf-8")
        assert "# OCR Output:" in content
        assert "lighton-ocr" in content
        assert "Hello World" in content

    def test_export_csv(self, tmp_path):
        exp = StructuredExporter(tmp_path, "csv")
        task = self._make_completed_task(tmp_path)
        out = exp.export_task(task, "glm-ocr", output_format="csv")
        assert out.suffix == ".csv"
        content = out.read_text(encoding="utf-8")
        assert "file_name" in content
        assert "invoice.pdf" in content

    def test_export_txt(self, tmp_path):
        exp = StructuredExporter(tmp_path, "txt")
        task = self._make_completed_task(tmp_path)
        out = exp.export_task(task, "glm-ocr", output_format="txt")
        assert out.suffix == ".txt"
        assert "Hello World" in out.read_text(encoding="utf-8")

    def test_export_manifest(self, tmp_path):
        exp = StructuredExporter(tmp_path, "json")
        tasks = [self._make_completed_task(tmp_path, f"doc{i}.pdf") for i in range(3)]
        for task in tasks:
            task.status = TaskStatus.COMPLETED
        manifest = exp.export_summary_manifest(tasks, "glm-ocr", elapsed_sec=5.5)
        assert manifest.exists()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["batch_summary"]["total_files"] == 3
        assert data["batch_summary"]["model_id"] == "glm-ocr"


# ── BatchTask.to_dict Tests ──────────────────────────────────────────────────

class TestBatchTask:
    def test_to_dict_contains_required_keys(self):
        task = _make_task("sample.pdf", "t-xyz")
        d = task.to_dict()
        for key in ("task_id", "source_path", "file_name", "status", "retries", "duration_sec"):
            assert key in d

    def test_to_dict_status_is_string(self):
        task = _make_task()
        task.status = TaskStatus.COMPLETED
        d = task.to_dict()
        assert d["status"] == "COMPLETED"

    def test_result_text_preview_truncated(self):
        task = _make_task()
        task.result_text = "A" * 300
        d = task.to_dict()
        assert len(d["result_text_preview"]) <= 153  # 150 + "..."


# ── JobMetrics Tests ─────────────────────────────────────────────────────────

class TestJobMetrics:
    def test_to_dict_all_fields_present(self):
        m = JobMetrics(
            status=BatchStatus.RUNNING,
            total_files=10,
            processed_files=5,
            failed_files=1,
        )
        d = m.to_dict()
        assert d["status"] == "RUNNING"
        assert d["total_files"] == 10
        assert d["processed_files"] == 5
        assert d["failed_files"] == 1


# ── BatchOCR Control API Tests ───────────────────────────────────────────────

class TestBatchOCRControls:
    def test_pause_sets_paused_status(self):
        from textlens.batch import BatchOCR
        batch = BatchOCR(enable_dashboard=False)
        batch._start_time = time.time()
        batch.pause()
        metrics = batch.get_metrics()
        assert metrics.status == BatchStatus.PAUSED

    def test_resume_restores_running_status(self):
        from textlens.batch import BatchOCR
        batch = BatchOCR(enable_dashboard=False)
        batch._start_time = time.time()
        batch.pause()
        batch.resume()
        metrics = batch.get_metrics()
        assert metrics.status == BatchStatus.RUNNING

    def test_cancel_sets_cancelled_status(self):
        from textlens.batch import BatchOCR
        batch = BatchOCR(enable_dashboard=False)
        batch._start_time = time.time()
        batch.cancel()
        metrics = batch.get_metrics()
        assert metrics.status == BatchStatus.CANCELLED

    def test_reconfigure_workers(self):
        from textlens.batch import BatchOCR
        batch = BatchOCR(workers=2, enable_dashboard=False)
        batch.reconfigure(workers=8)
        assert batch._config.workers == 8
        assert batch._metrics.target_workers == 8

    def test_reconfigure_output_format(self):
        from textlens.batch import BatchOCR
        batch = BatchOCR(output_format="json", enable_dashboard=False)
        batch.reconfigure(output_format="markdown")
        assert batch._config.output_format == "markdown"

    def test_get_logs_empty_initially(self):
        from textlens.batch import BatchOCR
        batch = BatchOCR(enable_dashboard=False)
        assert isinstance(batch.get_logs(), list)

    def test_retry_failed_on_empty_queue(self):
        from textlens.batch import BatchOCR
        batch = BatchOCR(enable_dashboard=False)
        count = batch.retry_failed()
        assert count == 0


# ── BatchOCR Discovery Tests ─────────────────────────────────────────────────

class TestBatchOCRDiscovery:
    def test_discovers_images_in_folder(self, tmp_path):
        from textlens.batch import BatchOCR
        (tmp_path / "a.png").write_bytes(b"fake")
        (tmp_path / "b.jpg").write_bytes(b"fake")
        (tmp_path / "skip.txt").write_bytes(b"txt")
        batch = BatchOCR(enable_dashboard=False)
        files = batch._discover_files(tmp_path)
        names = {f.name for f in files}
        assert "a.png" in names
        assert "b.jpg" in names
        assert "skip.txt" not in names

    def test_discovers_pdfs_in_folder(self, tmp_path):
        from textlens.batch import BatchOCR
        (tmp_path / "report.pdf").write_bytes(b"%PDF-fake")
        batch = BatchOCR(enable_dashboard=False)
        files = batch._discover_files(tmp_path)
        names = {f.name for f in files}
        assert "report.pdf" in names

    def test_discovers_recursively(self, tmp_path):
        from textlens.batch import BatchOCR
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.png").write_bytes(b"fake")
        batch = BatchOCR(enable_dashboard=False, recursive=True)
        files = batch._discover_files(tmp_path)
        names = {f.name for f in files}
        assert "deep.png" in names

    def test_no_recursive_flag(self, tmp_path):
        from textlens.batch import BatchOCR
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.png").write_bytes(b"fake")
        (tmp_path / "top.png").write_bytes(b"fake")
        batch = BatchOCR(enable_dashboard=False, recursive=False)
        files = batch._discover_files(tmp_path)
        names = {f.name for f in files}
        assert "top.png" in names
        assert "deep.png" not in names

    def test_empty_folder_returns_empty_list(self, tmp_path):
        from textlens.batch import BatchOCR
        batch = BatchOCR(enable_dashboard=False)
        files = batch._discover_files(tmp_path)
        assert files == []


# ── Dashboard Import Test ────────────────────────────────────────────────────

class TestDashboard:
    def test_dashboard_imports_without_error(self):
        from textlens.batch.dashboard import start_dashboard_server, _DASHBOARD_HTML
        assert "TextLens BatchOCR" in _DASHBOARD_HTML
        assert callable(start_dashboard_server)

    def test_dashboard_html_contains_sse_connect(self):
        from textlens.batch.dashboard import _DASHBOARD_HTML
        assert "EventSource" in _DASHBOARD_HTML
        assert "/api/stream" in _DASHBOARD_HTML

    def test_dashboard_html_contains_controls(self):
        from textlens.batch.dashboard import _DASHBOARD_HTML
        assert "/api/pause" in _DASHBOARD_HTML
        assert "/api/resume" in _DASHBOARD_HTML
        assert "/api/cancel" in _DASHBOARD_HTML
        assert "/api/retry-failed" in _DASHBOARD_HTML
        assert "/api/reconfigure" in _DASHBOARD_HTML
