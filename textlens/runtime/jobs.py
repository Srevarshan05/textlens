"""
textlens.runtime.jobs
─────────────────────
Asynchronous job execution with bounded concurrency and backpressure.

::

    submit() ──► bounded queue ──► N worker threads ──► JobStore
                    │ full? → CapacityError (HTTP 503 + Retry-After)
                    └─ cancel(), retries, progress, TTL eviction

The in-memory :class:`MemoryJobStore` needs no infrastructure.  The
:class:`JobStore` interface is the seam for a shared store (Redis, a SQL
table…) so API replicas and workers can be scaled independently — see
``docs/production/scaling.md``.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from textlens.errors import CancelledError, CapacityError, JobNotFoundError
from textlens.observability import metrics

logger = logging.getLogger("textlens.jobs")

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
TERMINAL = {SUCCEEDED, FAILED, CANCELLED}

_jobs_total = metrics.counter("textlens_jobs_total", "Jobs by final status")
_jobs_active = metrics.gauge("textlens_jobs_active", "Jobs queued or running")
_job_seconds = metrics.histogram("textlens_job_seconds", "Job wall time")


@dataclass
class Job:
    id: str
    kind: str = "ocr"
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempts: int = 0
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("Job was cancelled.")

    def to_dict(self, include_result: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempts": self.attempts,
            "progress": self.progress,
            "error": self.error,
            "meta": self.meta,
        }
        if self.started_at and self.finished_at:
            d["duration_s"] = round(self.finished_at - self.started_at, 3)
        if include_result:
            d["result"] = self.result
        return d


class JobStore(abc.ABC):
    """Where job state lives. Implement for Redis/SQL to share across replicas."""

    @abc.abstractmethod
    def put(self, job: Job) -> None: ...

    @abc.abstractmethod
    def get(self, job_id: str) -> Optional[Job]: ...

    @abc.abstractmethod
    def delete(self, job_id: str) -> None: ...

    @abc.abstractmethod
    def all(self) -> List[Job]: ...


class MemoryJobStore(JobStore):
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())


class JobManager:
    """Run callables as jobs with a concurrency cap and a queue limit.

    Parameters
    ----------
    workers:
        Concurrent jobs.  One per GPU is usually right for local VLMs; ONNX
        and remote backends tolerate more.
    max_queue:
        Jobs allowed to wait.  Beyond this :meth:`submit` raises
        :class:`~textlens.errors.CapacityError` (backpressure).
    ttl_seconds:
        Finished jobs are evicted after this long (privacy + memory).
    retries:
        Automatic retries for failures that are not input errors.
    """

    def __init__(self, workers: int = 1, max_queue: int = 64, ttl_seconds: float = 3600.0, retries: int = 0, store: Optional[JobStore] = None) -> None:
        self.workers = max(1, int(workers))
        self.max_queue = max(0, int(max_queue))
        self.ttl = ttl_seconds
        self.retries = retries
        self.store = store or MemoryJobStore()
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="textlens-job")
        self._lock = threading.Lock()
        self._pending = 0
        self._futures: Dict[str, Future] = {}
        self._closed = False

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    @property
    def saturated(self) -> bool:
        return self.pending >= self.workers + self.max_queue

    def submit(self, fn: Callable[[Job], Any], kind: str = "ocr", meta: Optional[Dict[str, Any]] = None) -> Job:
        self._evict()
        with self._lock:
            if self._closed:
                raise CapacityError("Server is shutting down.")
            if self._pending >= self.workers + self.max_queue:
                raise CapacityError(
                    f"Queue is full ({self._pending} jobs pending).",
                    hint="Retry later; the server applies backpressure instead of running out of memory.",
                    retry_after=5,
                )
            self._pending += 1
        job = Job(id=uuid.uuid4().hex, kind=kind, meta=dict(meta or {}))
        self.store.put(job)
        _jobs_active.inc()
        self._futures[job.id] = self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> Any:
        from textlens.errors import InputError

        try:
            while True:
                if job.cancel_event.is_set():
                    job.status = CANCELLED
                    return None
                job.status = RUNNING
                job.started_at = job.started_at or time.time()
                job.attempts += 1
                self.store.put(job)
                try:
                    job.result = fn(job)
                    job.status = SUCCEEDED
                    return job.result
                except CancelledError:
                    job.status = CANCELLED
                    return None
                except Exception as exc:  # noqa: BLE001 - recorded on the job
                    retryable = not isinstance(exc, InputError) and job.attempts <= self.retries
                    job.error = {
                        "error": getattr(exc, "code", type(exc).__name__),
                        "message": getattr(exc, "message", None) or str(exc),
                        "hint": getattr(exc, "hint", None),
                    }
                    if retryable:
                        logger.warning("Job %s attempt %d failed (%s); retrying", job.id, job.attempts, exc)
                        time.sleep(min(10.0, 0.5 * 2 ** (job.attempts - 1)))
                        continue
                    logger.debug("Job %s failed:\n%s", job.id, traceback.format_exc())
                    job.status = FAILED
                    return None
        finally:
            job.finished_at = time.time()
            self.store.put(job)
            with self._lock:
                self._pending -= 1
            self._futures.pop(job.id, None)
            _jobs_active.dec()
            _jobs_total.inc(status=job.status, kind=job.kind)
            if job.started_at:
                _job_seconds.observe(job.finished_at - job.started_at, kind=job.kind)

    def get(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise JobNotFoundError(f"No job with id {job_id!r} (it may have expired).")
        return job

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Job:
        fut = self._futures.get(job_id)
        if fut is not None:
            fut.result(timeout=timeout)
        return self.get(job_id)

    def future(self, job_id: str) -> Optional[Future]:
        return self._futures.get(job_id)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status not in TERMINAL:
            job.cancel_event.set()
            fut = self._futures.get(job_id)
            if fut is not None and fut.cancel():  # never started
                job.status = CANCELLED
                job.finished_at = time.time()
                with self._lock:
                    self._pending -= 1
                _jobs_active.dec()
            self.store.put(job)
        return job

    def _evict(self) -> None:
        if not self.ttl:
            return
        cutoff = time.time() - self.ttl
        for job in self.store.all():
            if job.status in TERMINAL and job.finished_at and job.finished_at < cutoff:
                self.store.delete(job.id)

    def stats(self) -> Dict[str, Any]:
        jobs = self.store.all()
        counts: Dict[str, int] = {}
        for j in jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        return {"workers": self.workers, "max_queue": self.max_queue, "pending": self.pending, "jobs": counts}

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        for job in self.store.all():
            if job.status not in TERMINAL:
                job.cancel_event.set()
        self._pool.shutdown(wait=wait, cancel_futures=True)
