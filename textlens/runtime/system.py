"""
textlens.runtime.system
───────────────────────
What machine are we on, and which inference runtimes can we use?

Everything here is cheap: GPU facts come from ``nvidia-smi`` (see
:mod:`textlens.models.hardware`), runtime versions from package metadata.
PyTorch is **not** imported — importing it can take seconds and allocate
GPU memory.  ``inspect_system(deep=True)`` opts into an actual
``torch.cuda.is_available()`` probe for ``textlens doctor --deep``.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import os
import platform
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from textlens.models.hardware import HardwareProfile, inspect_hardware


def _version(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclasses.dataclass(frozen=True)
class PlatformInfo:
    system: str
    machine: str
    python: str
    device_class: str  # desktop | raspberry-pi | jetson | apple-silicon | arm | server
    board: Optional[str] = None  # e.g. "Raspberry Pi 5 Model B Rev 1.0"
    l4t: Optional[str] = None  # Jetson Linux (L4T) release
    container: bool = False

    @property
    def is_edge(self) -> bool:
        return self.device_class in ("raspberry-pi", "jetson", "arm")


@dataclasses.dataclass(frozen=True)
class RuntimeInfo:
    onnxruntime: Optional[str] = None
    ort_providers: Tuple[str, ...] = ()
    torch: Optional[str] = None
    torch_cuda_build: Optional[bool] = None
    torch_cuda_available: Optional[bool] = None  # only set by a deep probe
    transformers: Optional[str] = None
    accelerate: Optional[str] = None
    huggingface_hub: Optional[str] = None
    vllm: Optional[str] = None
    tensorrt: Optional[str] = None
    paddle: Optional[str] = None
    fastapi: Optional[str] = None
    fast_plate_ocr: Optional[str] = None
    open_image_models: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    hardware: HardwareProfile
    platform: PlatformInfo
    runtimes: RuntimeInfo

    @property
    def gpu_present(self) -> bool:
        return bool(self.hardware.gpus)

    @property
    def vram_gb(self) -> float:
        return float(self.hardware.primary_vram_gb or 0.0)

    @property
    def ram_gb(self) -> float:
        return float(self.hardware.ram_total_gb or 0.0)

    @property
    def torch_cuda(self) -> bool:
        """Can PyTorch models use the GPU (best estimate without importing torch)?"""
        rt = self.runtimes
        if rt.torch_cuda_available is not None:
            return rt.torch_cuda_available
        return bool(self.gpu_present and rt.torch and rt.torch_cuda_build)

    @property
    def ort_cuda(self) -> bool:
        return "CUDAExecutionProvider" in self.runtimes.ort_providers and self.gpu_present

    @property
    def is_edge(self) -> bool:
        return self.platform.is_edge

    def to_dict(self) -> Dict[str, object]:
        hw = dataclasses.asdict(self.hardware)
        return {
            "platform": dataclasses.asdict(self.platform),
            "hardware": hw,
            "runtimes": self.runtimes.to_dict(),
            "torch_cuda": self.torch_cuda,
            "ort_cuda": self.ort_cuda,
        }


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(errors="ignore").strip("\x00").strip()
    except OSError:
        return None


def detect_platform() -> PlatformInfo:
    system = platform.system()
    machine = platform.machine().lower()
    board = _read("/proc/device-tree/model") if system == "Linux" else None
    l4t = None
    if system == "Linux":
        rel = _read("/etc/nv_tegra_release")
        if rel:
            l4t = rel.split(",")[0].replace("# ", "").strip()
    if board and "raspberry pi" in board.lower():
        device_class = "raspberry-pi"
    elif l4t or (board and "jetson" in board.lower()):
        device_class = "jetson"
    elif system == "Darwin" and machine in ("arm64", "aarch64"):
        device_class = "apple-silicon"
    elif machine in ("aarch64", "arm64", "armv7l", "armv6l"):
        device_class = "arm"
    else:
        device_class = "desktop"
    container = os.path.exists("/.dockerenv") or bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    return PlatformInfo(
        system=system,
        machine=machine,
        python=platform.python_version(),
        device_class=device_class,
        board=board,
        l4t=l4t,
        container=container,
    )


def _torch_cuda_build(version: Optional[str], system: str) -> Optional[bool]:
    if not version:
        return None
    v = version.lower()
    if "+cpu" in v:
        return False
    if "+cu" in v or "nv" in v or "+rocm" in v:
        return True
    # PyPI Linux wheels are CUDA builds without a local tag and pull in the
    # nvidia-* runtime wheels; Windows/macOS PyPI wheels are CPU-only.
    if system == "Linux":
        return any(_version(d) for d in ("nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime"))
    return False


def detect_runtimes(system: Optional[str] = None, deep: bool = False) -> RuntimeInfo:
    system = system or platform.system()
    ort_version = None
    providers: Tuple[str, ...] = ()
    for dist in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"):
        ort_version = ort_version or _version(dist)
    if ort_version and _has_module("onnxruntime"):
        try:
            import onnxruntime

            providers = tuple(onnxruntime.get_available_providers())
        except Exception:
            providers = ()
    torch_v = _version("torch")
    cuda_available = None
    if deep and torch_v:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    return RuntimeInfo(
        onnxruntime=ort_version,
        ort_providers=providers,
        torch=torch_v,
        torch_cuda_build=_torch_cuda_build(torch_v, system),
        torch_cuda_available=cuda_available,
        transformers=_version("transformers"),
        accelerate=_version("accelerate"),
        huggingface_hub=_version("huggingface_hub") or _version("huggingface-hub"),
        vllm=_version("vllm"),
        tensorrt=_version("tensorrt"),
        paddle=_version("paddlepaddle") or _version("paddlepaddle-gpu"),
        fastapi=_version("fastapi"),
        fast_plate_ocr=_version("fast-plate-ocr"),
        open_image_models=_version("open-image-models"),
    )


_lock = threading.Lock()
_cached: Optional[SystemInfo] = None


def inspect_system(refresh: bool = False, deep: bool = False) -> SystemInfo:
    """Return (and cache) a snapshot of hardware, platform and runtimes."""
    global _cached
    with _lock:
        if _cached is not None and not refresh and not deep:
            return _cached
        plat = detect_platform()
        info = SystemInfo(hardware=inspect_hardware(), platform=plat, runtimes=detect_runtimes(plat.system, deep=deep))
        _cached = info
        return info


def set_system_info(info: Optional[SystemInfo]) -> None:
    """Override the cached snapshot (tests, simulations)."""
    global _cached
    with _lock:
        _cached = info
