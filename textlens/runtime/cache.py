"""
textlens.runtime.cache
──────────────────────
Result cache keyed by *what was processed* and *how*.

``key = sha256(document content hash + pipeline configuration hash)``

so the same bytes processed with the same model/profile/options return
instantly, while any change (different model, pages, DPI, TextLens version)
misses.  Modes:

``off``     no caching
``memory``  process-local LRU (default)
``disk``    LRU + gzip JSON files under ``TEXTLENS_CACHE_DIR`` (survives restarts)

Disk caching stores recognised text, so it is opt-in and disabled entirely
by ``TEXTLENS_NO_PERSIST=1``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("textlens.cache")


def config_hash(config: Dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class ResultCache:
    def __init__(self, mode: Optional[str] = None, directory: Optional[Path] = None, max_items: int = 256, ttl_seconds: Optional[float] = None) -> None:
        from textlens.config import get_settings

        settings = get_settings()
        mode = (mode or settings.result_cache).lower()
        if settings.no_persist and mode == "disk":
            mode = "memory"
        self.mode = mode
        self.directory = Path(directory or settings.results_dir)
        self.max_items = max_items
        self.ttl = ttl_seconds
        self._mem: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(document_hash: str, cfg_hash: str) -> str:
        return hashlib.sha256(f"{document_hash}:{cfg_hash}".encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json.gz"

    def get(self, key: str) -> Optional[Any]:
        if self.mode == "off":
            return None
        from textlens.core.result import Result

        now = time.time()
        with self._lock:
            hit = self._mem.get(key)
            if hit is not None:
                stamp, data = hit
                if self.ttl is None or now - stamp <= self.ttl:
                    self._mem.move_to_end(key)
                    self.hits += 1
                    return Result.from_dict(data)
                self._mem.pop(key, None)
        if self.mode == "disk":
            path = self._path(key)
            try:
                if self.ttl is not None and now - path.stat().st_mtime > self.ttl:
                    raise FileNotFoundError
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._remember(key, data)
                self.hits += 1
                return Result.from_dict(data)
            except (OSError, ValueError):
                pass
        self.misses += 1
        return None

    def _remember(self, key: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._mem[key] = (time.time(), data)
            self._mem.move_to_end(key)
            while len(self._mem) > self.max_items:
                self._mem.popitem(last=False)

    def put(self, key: str, result: Any) -> None:
        if self.mode == "off":
            return
        data = result.to_dict()
        self._remember(key, data)
        if self.mode == "disk":
            path = self._path(key)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False)
                os.replace(tmp, path)
            except OSError as exc:
                logger.warning("Could not write result cache %s: %s", path, exc)

    def clear(self) -> int:
        with self._lock:
            n = len(self._mem)
            self._mem.clear()
        if self.mode == "disk" and self.directory.exists():
            for p in self.directory.rglob("*.json.gz"):
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        return n

    def stats(self) -> Dict[str, Any]:
        return {"mode": self.mode, "items": len(self._mem), "hits": self.hits, "misses": self.misses}


_shared: Dict[str, ResultCache] = {}
_shared_lock = threading.Lock()


def shared_cache(mode: Optional[str] = None) -> ResultCache:
    """Process-wide cache instance per mode."""
    from textlens.config import get_settings

    m = (mode or get_settings().result_cache).lower()
    with _shared_lock:
        if m not in _shared:
            _shared[m] = ResultCache(m)
        return _shared[m]
