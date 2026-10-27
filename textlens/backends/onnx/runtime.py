"""
textlens.backends.onnx.runtime
──────────────────────────────
ONNX Runtime session factory for edge and CPU deployments.

Execution providers are chosen automatically:

=====================  =====================================================
``device``             providers tried (first available wins)
=====================  =====================================================
``None`` / ``auto``    CUDA → DirectML → CPU (TensorRT / CoreML are opt-in)
``cpu``                CPU
``cuda`` / ``cuda:N``  TensorRT (if ``TEXTLENS_ORT_TENSORRT=1``) → CUDA → CPU
``tensorrt``           TensorRT → CUDA → CPU   (Jetson: needs NVIDIA's ORT wheel)
``coreml``             CoreML → CPU            (Apple Silicon)
``dml``                DirectML → CPU          (Windows, onnxruntime-directml)
=====================  =====================================================

TensorRT engines are cached under ``$TEXTLENS_HOME/trt-cache`` so the slow
first build happens once per model and device.

``low_memory=True`` disables ORT's CPU memory arena: a smaller resident
footprint for ≤1 GB boards, at a measured ~3× detection slowdown.  Use it
only when memory, not latency, is the constraint.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple, Union

from textlens.errors import BackendUnavailableError

logger = logging.getLogger("textlens.onnx")

Provider = Union[str, Tuple[str, dict]]


def ort():
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:  # pragma: no cover - core dependency
        raise BackendUnavailableError(
            "onnxruntime is not installed.",
            hint="pip install onnxruntime  (Jetson: install NVIDIA's onnxruntime-gpu wheel for JetPack)",
        ) from exc
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)
    return onnxruntime


def available_providers() -> List[str]:
    return list(ort().get_available_providers())


def _device_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def select_providers(device: Optional[str] = None) -> List[Provider]:
    """Return an ordered provider list for ``InferenceSession``."""
    from textlens.config import get_settings

    avail = set(available_providers())
    dev = (device or "auto").lower()
    idx = _device_index(dev)
    trt_cache = str(get_settings().home / "trt-cache")
    want_trt = dev.startswith("tensorrt") or os.environ.get("TEXTLENS_ORT_TENSORRT", "") == "1"
    chosen: List[Provider] = []

    def add(name: str, opts: Optional[dict] = None) -> None:
        if name in avail:
            chosen.append((name, opts) if opts else name)

    if dev == "cpu":
        pass
    elif dev.startswith(("cuda", "gpu", "tensorrt", "auto")):
        if want_trt:
            add("TensorrtExecutionProvider", {"device_id": idx, "trt_engine_cache_enable": True, "trt_engine_cache_path": trt_cache, "trt_fp16_enable": True})
        add("CUDAExecutionProvider", {"device_id": idx})
        if dev == "auto":
            add("DmlExecutionProvider")
    elif dev == "coreml":
        add("CoreMLExecutionProvider")
    elif dev in ("dml", "directml"):
        add("DmlExecutionProvider", {"device_id": idx})
    chosen.append("CPUExecutionProvider")
    return chosen


def make_session(
    model_path: str,
    device: Optional[str] = None,
    threads: Optional[int] = None,
    low_memory: bool = False,
) -> Tuple[Any, str]:
    """Create an ``InferenceSession``; returns ``(session, active_provider)``."""
    rt = ort()
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    # ORT's thread pools busy-spin after each run by default.  With several
    # sessions (detector + recogniser) that steals the CPU from PDF
    # rendering and the other stages of the pipeline; measured 2–8× slowdowns
    # on repeated runs.  Threads sleep instead.
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    so.intra_op_num_threads = int(threads) if threads else _default_threads()
    so.inter_op_num_threads = 1
    # OCR inputs change shape every call (page size, batch width); memory
    # pattern planning only pays off for fixed shapes.
    so.enable_mem_pattern = False
    if low_memory:
        so.enable_cpu_mem_arena = False
    providers = select_providers(device)
    try:
        session = rt.InferenceSession(model_path, sess_options=so, providers=providers)
    except Exception as exc:
        logger.warning("Falling back to CPU for %s: %s", os.path.basename(model_path), exc)
        session = rt.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    active = session.get_providers()[0]
    return session, active


def _default_threads() -> int:
    """Physical cores (hyper-threads rarely help conv inference), max 8."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        cores = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(8, int(cores)))


def provider_to_device(provider: str) -> str:
    return {
        "CPUExecutionProvider": "cpu",
        "CUDAExecutionProvider": "cuda",
        "TensorrtExecutionProvider": "tensorrt",
        "DmlExecutionProvider": "directml",
        "CoreMLExecutionProvider": "coreml",
    }.get(provider, provider)
