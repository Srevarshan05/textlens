"""
textlens.backends.loader
────────────────────────
Resolve a model spec to a live backend, lazily and exactly once.

* Runtime requirements are checked with ``importlib.util.find_spec`` —
  nothing heavy is imported until a backend is really used.
* Missing runtimes raise :class:`~textlens.errors.BackendUnavailableError`
  with the exact ``pip install`` command.
* Loaded backends live in a process-wide :class:`BackendPool`, keyed by
  model, device and options, so every ``OCR`` instance, batch worker and
  server request shares one copy of the weights instead of one per thread.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from textlens.backends.base import OCRBackend
from textlens.errors import BackendLoadError, BackendUnavailableError
from textlens.models.specs import STATUS_CATALOG, ModelSpec

logger = logging.getLogger("textlens.backends.loader")


def missing_requirements(spec: ModelSpec) -> List[str]:
    """Importable module names required by *spec* that are absent."""
    missing = []
    for mod in spec.requires:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    return missing


def install_hint(spec: ModelSpec) -> str:
    if spec.extra:
        return f'pip install "textlens-ocr[{spec.extra}]"'
    missing = missing_requirements(spec)
    return f"pip install {' '.join(missing)}" if missing else ""


def _import_adapter(path: str) -> type:
    module_name, _, cls_name = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def create_backend(spec: ModelSpec, device: Optional[str] = None, **options: Any) -> OCRBackend:
    """Instantiate (but do not load) the adapter for *spec*."""
    if spec.status == STATUS_CATALOG or not spec.adapter:
        raise BackendUnavailableError(
            f"{spec.display_name} is a catalog entry without a bundled adapter.",
            hint=spec.license_notes or f"See {spec.source_url} for how to run it; serve it behind an OpenAI-compatible API to use it from TextLens.",
            model=spec.id,
        )
    missing = missing_requirements(spec)
    if missing:
        raise BackendUnavailableError(
            f"{spec.display_name} needs {', '.join(missing)}, which is not installed.",
            hint=install_hint(spec),
            model=spec.id,
            missing=missing,
        )
    try:
        cls = _import_adapter(spec.adapter)
    except Exception as exc:
        raise BackendLoadError(f"Cannot import adapter {spec.adapter}: {exc}", model=spec.id) from exc
    return cls(spec, device=device, **options)


class BackendPool:
    """Process-wide cache of backends: one loaded copy per (model, device, options)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backends: Dict[Tuple[str, Optional[str], str], OCRBackend] = {}

    @staticmethod
    def _key(spec: ModelSpec, device: Optional[str], options: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
        return (spec.id, device, json.dumps(options, sort_keys=True, default=str))

    def get(self, spec: ModelSpec, device: Optional[str] = None, **options: Any) -> OCRBackend:
        key = self._key(spec, device, options)
        with self._lock:
            backend = self._backends.get(key)
            if backend is None:
                backend = create_backend(spec, device=device, **options)
                self._backends[key] = backend
            return backend

    def loaded(self) -> List[OCRBackend]:
        with self._lock:
            return [b for b in self._backends.values() if b.is_loaded]

    def unload(self, model_id: Optional[str] = None) -> int:
        """Unload one model (or all) and free its memory; returns count."""
        with self._lock:
            victims = [k for k in self._backends if model_id is None or k[0] == model_id]
            backends = [self._backends.pop(k) for k in victims]
        for b in backends:
            try:
                b.unload()
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Error unloading %s: %s", b.name, exc)
        return len(backends)


_pool = BackendPool()


def get_pool() -> BackendPool:
    return _pool


def get_backend(spec: ModelSpec, device: Optional[str] = None, **options: Any) -> OCRBackend:
    """Shared backend for *spec* from the process-wide pool."""
    return _pool.get(spec, device=device, **options)
