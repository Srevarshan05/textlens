"""
textlens.observability
──────────────────────
Structured logging and in-process metrics with zero extra dependencies.

Logging
    ``configure_logging()`` installs a handler on the ``textlens`` logger
    only (never the root logger).  ``TEXTLENS_LOG_FORMAT=json`` emits one
    JSON object per line — ready for Loki, ELK, CloudWatch or Datadog.

Metrics
    :data:`metrics` is a tiny Prometheus-compatible registry (counters,
    gauges, histograms).  The server exposes it at ``GET /metrics``; batch
    jobs and the CLI profiler read it directly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


class JsonFormatter(logging.Formatter):
    _SKIP = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info",
        "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread",
        "threadName", "processName", "process", "taskName", "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


_configured = False


def configure_logging(level: Optional[int] = None, fmt: Optional[str] = None) -> logging.Logger:
    """Attach one handler to the ``textlens`` logger (idempotent)."""
    global _configured
    from textlens.config import get_settings

    settings = get_settings()
    logger = logging.getLogger("textlens")
    if level is None:
        level = getattr(logging, settings.log_level, logging.WARNING)
    logger.setLevel(level)
    if not _configured:
        handler = logging.StreamHandler()
        if (fmt or settings.log_format) == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True
    return logger


# ── metrics ──────────────────────────────────────────────────────────────────

_DEFAULT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
LabelKey = Tuple[Tuple[str, str], ...]


class _Metric:
    def __init__(self, name: str, help_: str, kind: str) -> None:
        self.name, self.help, self.kind = name, help_, kind
        self.lock = threading.Lock()


class Counter(_Metric):
    def __init__(self, name: str, help_: str) -> None:
        super().__init__(name, help_, "counter")
        self.values: Dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self.lock:
            self.values[key] = self.values.get(key, 0.0) + amount


class Gauge(_Metric):
    def __init__(self, name: str, help_: str) -> None:
        super().__init__(name, help_, "gauge")
        self.values: Dict[LabelKey, float] = {}

    def set(self, value: float, **labels: str) -> None:
        with self.lock:
            self.values[tuple(sorted(labels.items()))] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self.lock:
            self.values[key] = self.values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)


class Histogram(_Metric):
    def __init__(self, name: str, help_: str, buckets: Sequence[float] = _DEFAULT_BUCKETS) -> None:
        super().__init__(name, help_, "histogram")
        self.buckets = tuple(buckets)
        self.data: Dict[LabelKey, List[float]] = {}  # bucket counts + [sum, count]

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self.lock:
            row = self.data.setdefault(key, [0.0] * (len(self.buckets) + 2))
            for i, b in enumerate(self.buckets):
                if value <= b:
                    row[i] += 1
            row[-2] += value
            row[-1] += 1

    @contextmanager
    def time(self, **labels: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - t0, **labels)


def _fmt_labels(key: LabelKey, extra: Optional[Tuple[str, str]] = None) -> str:
    items = list(key) + ([extra] if extra else [])
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{str(v).replace(chr(34), chr(39))}"' for k, v in items) + "}"


class Registry:
    def __init__(self) -> None:
        self._metrics: Dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _get(self, cls, name: str, help_: str, **kw):  # type: ignore[no-untyped-def]
        with self._lock:
            m = self._metrics.get(name)
            if m is None:
                m = cls(name, help_, **kw)
                self._metrics[name] = m
            return m

    def counter(self, name: str, help_: str = "") -> Counter:
        return self._get(Counter, name, help_)

    def gauge(self, name: str, help_: str = "") -> Gauge:
        return self._get(Gauge, name, help_)

    def histogram(self, name: str, help_: str = "", buckets: Sequence[float] = _DEFAULT_BUCKETS) -> Histogram:
        return self._get(Histogram, name, help_, buckets=buckets)

    def render(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        lines: List[str] = []
        with self._lock:
            metrics = list(self._metrics.values())
        for m in metrics:
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} {m.kind}")
            with m.lock:
                if isinstance(m, (Counter, Gauge)):
                    for key, v in m.values.items():
                        lines.append(f"{m.name}{_fmt_labels(key)} {v:g}")
                elif isinstance(m, Histogram):
                    for key, row in m.data.items():
                        for b, c in zip(m.buckets, row):
                            lines.append(f"{m.name}_bucket{_fmt_labels(key, ('le', f'{b:g}'))} {c:g}")
                        lines.append(f"{m.name}_bucket{_fmt_labels(key, ('le', '+Inf'))} {row[-1]:g}")
                        lines.append(f"{m.name}_sum{_fmt_labels(key)} {row[-2]:g}")
                        lines.append(f"{m.name}_count{_fmt_labels(key)} {row[-1]:g}")
        return "\n".join(lines) + "\n"


metrics = Registry()
