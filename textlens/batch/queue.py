"""
textlens.batch.queue
────────────────────
Pluggable queue backends for BatchOCR tasks.

Initial implementation provides `MemoryBatchQueue` powered by Python's
thread-safe `queue.Queue`. The abstract `BaseBatchQueue` interface allows
future seamless integration of Redis, RabbitMQ, or distributed queues without
altering the public API.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from textlens.batch.types import BatchTask, TaskStatus


class BaseBatchQueue(ABC):
    """Abstract base class for batch task queues."""

    @abstractmethod
    def enqueue(self, task: BatchTask) -> None:
        """Add a task to the queue."""
        pass

    @abstractmethod
    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[BatchTask]:
        """Fetch the next task from the queue."""
        pass

    @abstractmethod
    def task_done(self, task: BatchTask) -> None:
        """Mark a task as completed/processed."""
        pass

    @abstractmethod
    def requeue_failed(self, task: BatchTask) -> bool:
        """Re-queue a failed task for retry if remaining retries exist."""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[BatchTask]:
        """Retrieve task metadata by task ID."""
        pass

    @abstractmethod
    def get_all_tasks(self) -> List[BatchTask]:
        """Return a snapshot list of all tasks."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return current count of pending tasks in queue."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all pending tasks."""
        pass


class MemoryBatchQueue(BaseBatchQueue):
    """Thread-safe standard-library queue implementation for BatchOCR."""

    def __init__(self) -> None:
        self._queue: queue.Queue[BatchTask] = queue.Queue()
        self._tasks: Dict[str, BatchTask] = {}
        self._lock = threading.RLock()

    def enqueue(self, task: BatchTask) -> None:
        with self._lock:
            task.status = TaskStatus.QUEUED
            self._tasks[task.task_id] = task
            self._queue.put(task)

    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[BatchTask]:
        try:
            task = self._queue.get(block=block, timeout=timeout)
            with self._lock:
                task.status = TaskStatus.PROCESSING
                task.started_at = task.started_at or task.created_at
            return task
        except queue.Empty:
            return None

    def task_done(self, task: BatchTask) -> None:
        with self._lock:
            self._tasks[task.task_id] = task
            try:
                self._queue.task_done()
            except ValueError:
                pass

    def requeue_failed(self, task: BatchTask) -> bool:
        with self._lock:
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.RETRYING
                task.error = f"Retrying attempt {task.retries}/{task.max_retries}: {task.error or 'Unknown error'}"
                self._tasks[task.task_id] = task
                self._queue.put(task)
                return True
            else:
                task.status = TaskStatus.FAILED
                self._tasks[task.task_id] = task
                try:
                    self._queue.task_done()
                except ValueError:
                    pass
                return False

    def get_task(self, task_id: str) -> Optional[BatchTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[BatchTask]:
        with self._lock:
            return list(self._tasks.values())

    def size(self) -> int:
        return self._queue.qsize()

    def clear(self) -> None:
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except (queue.Empty, ValueError):
                    break
            self._tasks.clear()
