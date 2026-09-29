"""
textlens.runtime.batching
─────────────────────────
Dynamic batching and memory-aware batch sizing.

:class:`DynamicBatcher` implements the *collector* pattern used by
production OCR services: a worker blocks for the first request, then keeps
collecting until either ``max_batch`` items arrive or ``window_ms`` elapses,
and runs the whole batch through one ``backend.recognize(images)`` call.
Callers get a :class:`~concurrent.futures.Future` per item, so request
handlers stay simple while the GPU sees batches.

Only one batch runs at a time per batcher: this keeps GPU memory
predictable (the batch size, not the request rate, bounds VRAM).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger("textlens.batching")


@dataclass
class _Item:
    payload: Any
    future: Future
    enqueued: float


class DynamicBatcher:
    """Group concurrent requests into batches for one processing function.

    Parameters
    ----------
    process:
        ``process(payloads) -> results`` (same length and order).
    max_batch:
        Upper bound on items per call (see :func:`suggest_batch_size`).
    window_ms:
        How long to wait for more items after the first arrives.
    max_queue:
        Items allowed to wait; ``submit`` raises when full (backpressure).
    """

    def __init__(self, process: Callable[[List[Any]], Sequence[Any]], max_batch: int = 8, window_ms: float = 20.0, max_queue: int = 256, name: str = "batcher") -> None:
        self._process = process
        self.max_batch = max(1, int(max_batch))
        self.window = max(0.0, window_ms / 1000.0)
        self._queue: "queue.Queue[Optional[_Item]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.batches = 0
        self.items = 0
        self._thread = threading.Thread(target=self._loop, name=f"textlens-{name}", daemon=True)
        self._thread.start()

    def submit(self, payload: Any) -> Future:
        from textlens.errors import CapacityError

        fut: Future = Future()
        try:
            self._queue.put_nowait(_Item(payload, fut, time.perf_counter()))
        except queue.Full as exc:
            raise CapacityError("Batch queue is full.", retry_after=1) from exc
        return fut

    def __call__(self, payload: Any, timeout: Optional[float] = None) -> Any:
        return self.submit(payload).result(timeout=timeout)

    def _collect(self) -> List[_Item]:
        first = self._queue.get()
        if first is None:
            return []
        batch = [first]
        deadline = time.perf_counter() + self.window
        while len(batch) < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                self._stop.set()
                break
            batch.append(item)
        return batch

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                break
            live = [it for it in batch if not it.future.cancelled()]
            if not live:
                continue
            try:
                results = list(self._process([it.payload for it in live]))
                if len(results) != len(live):
                    raise RuntimeError(f"batch function returned {len(results)} results for {len(live)} inputs")
                for it, res in zip(live, results):
                    it.future.set_result(res)
            except Exception as exc:  # noqa: BLE001 - delivered to every caller
                for it in live:
                    if not it.future.done():
                        it.future.set_exception(exc)
            self.batches += 1
            self.items += len(live)

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)

    @property
    def mean_batch_size(self) -> float:
        return self.items / self.batches if self.batches else 0.0


def suggest_batch_size(spec: Any, system: Any = None, per_item_gb: Optional[float] = None, cap: int = 32) -> Tuple[int, str]:
    """Memory-aware batch size for a model on this machine.

    Uses free VRAM (GPU) or RAM (CPU) minus the model's footprint, divided
    by a per-item activation estimate.  Returns ``(size, explanation)``.
    Estimates are conservative; measure with ``textlens profile``.
    """
    if system is None:
        from textlens.runtime.system import inspect_system

        system = inspect_system()
    hw = system.hardware
    if spec.backend == "openai":
        return min(cap, 16), "remote server batches continuously; limited by client concurrency"
    gpu = spec.backend == "transformers" and system.torch_cuda and hw.gpus
    if gpu:
        free = max((g.vram_free_gb for g in hw.gpus), default=hw.primary_vram_gb)
        model_gb = spec.min_vram_gb or 2.0
        item = per_item_gb or max(0.25, 0.15 * model_gb)
        size = int(max(0.0, free - model_gb) // item)
        return max(1, min(cap, size)), f"{free:.1f} GB free VRAM − {model_gb:g} GB model ≈ {size} × {item:.2f} GB/page"
    ram = hw.ram_total_gb or 4.0
    item = per_item_gb or (0.15 if spec.backend == "onnx" else 1.0)
    budget = max(0.5, ram * 0.25)
    size = int(budget // item)
    cores = max(1, hw.cpu_physical_cores or 1)
    size = min(size, cores * 2)
    return max(1, min(cap, size)), f"25% of {ram:.0f} GB RAM at ~{item:g} GB/page, ≤ 2×{cores} cores"
