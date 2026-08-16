"""
textlens.models.hardware
─────────────────────────
Real hardware introspection — NO AI, NO guessing.

Collects factual system information using only:
- ``torch``      : CUDA availability, GPU name, VRAM, torch version
- ``platform``   : OS name and version
- ``psutil``     : CPU name, physical/logical cores, RAM
- ``GPUtil``     : GPU name/VRAM as a fallback when torch CUDA is not built
- ``subprocess`` : nvidia-smi for GPU names on systems without torch CUDA

All fields are detected at call time and returned as an immutable dataclass.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import logging
import os
import platform
import re
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger("textlens.models.hardware")


# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

# Importing PyTorch can take several seconds on a cold Windows process.  The
# Doctor and model advisor only need the driver-reported hardware, so defer the
# import until a legacy caller explicitly asks for torch-level inspection.
torch = None
_TORCH: Optional[bool] = None


def _get_torch():
    """Return PyTorch on demand, or ``None`` when it is unavailable."""
    global _TORCH, torch
    if _TORCH is None:
        try:
            import torch as imported_torch

            torch = imported_torch
            _TORCH = True
        except ImportError:
            _TORCH = False
    return torch if _TORCH else None


def _installed_torch_version() -> str:
    """Return the installed torch distribution version without importing it."""
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return "Not Installed"

try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import GPUtil  # type: ignore[import]

    _GPUTIL = True
except ImportError:
    _GPUTIL = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    """Hardware details for a single GPU device."""

    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_available: bool


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    """Full system hardware profile collected by :func:`inspect_hardware`.

    Attributes
    ----------
    os_name : str
        Operating system name + version (e.g. ``"Windows 11"``).
    python_version : str
        Python version string (e.g. ``"3.12.5"``).
    torch_version : str
        PyTorch version or ``"Not Installed"``.
    cuda_available : bool
        Whether PyTorch can use CUDA.
    cuda_version : str, optional
        CUDA toolkit version compiled into PyTorch (e.g. ``"12.8"``).
    system_cuda_version : str, optional
        CUDA driver version from nvidia-smi (e.g. ``"12.8"``).
    gpus : list[GPUInfo]
        List of detected GPU devices (may be empty on CPU-only systems).
    primary_gpu_name : str, optional
        Name of the first (primary) GPU.
    primary_vram_gb : float
        VRAM of the primary GPU in gigabytes (``0.0`` for CPU-only systems).
    cpu_name : str
        CPU model name.
    cpu_physical_cores : int
        Number of physical CPU cores.
    cpu_logical_cores : int
        Number of logical CPU cores (threads).
    ram_total_gb : float
        Total system RAM in gigabytes.
    device_type : str
        ``"cuda"`` or ``"cpu"``.
    """

    os_name: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    system_cuda_version: Optional[str]
    gpus: List[GPUInfo]
    primary_gpu_name: Optional[str]
    primary_vram_gb: float
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_total_gb: float
    device_type: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_os() -> str:
    """Return a human-readable OS name (e.g. ``"Windows 11"``)."""
    system = platform.system()
    if system == "Windows":
        release = platform.release()
        try:
            version = platform.version()
            # Windows 11 reports major build ≥ 22000
            build = int(version.split(".")[2])
            if build >= 22000:
                return "Windows 11"
        except Exception:
            pass
        return f"Windows {release}"
    elif system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}" if mac_ver else "macOS"
    elif system == "Linux":
        try:
            import distro  # type: ignore[import]

            return distro.name(pretty=True)
        except ImportError:
            return f"Linux {platform.release()}"
    return system


def _detect_cpu_name() -> str:
    """Return CPU model name using psutil or platform fallback."""
    system = platform.system()
    try:
        if system == "Windows":
            # WMIC was removed from recent Windows releases.  Resolve an
            # actual executable first so PowerShell cannot invoke its `cpu`
            # alias and pollute the CLI with an error message.
            from shutil import which

            wmic = which("wmic.exe") or which("wmic")
            if not wmic:
                return platform.processor() or "Unknown CPU"
            out = subprocess.check_output(
                [wmic, "cpu", "get", "Name"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
            )
            lines = [
                line.strip()
                for line in out.splitlines()
                if line.strip() and line.strip() != "Name"
            ]
            if lines:
                return lines[0]
        elif system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            )
            return out.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _detect_ram_gb() -> float:
    """Return total system RAM in GB."""
    if _PSUTIL:
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    # Fallback
    return 0.0


def _detect_cpu_cores() -> tuple[int, int]:
    """Return (physical_cores, logical_cores)."""
    if _PSUTIL:
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or 1
        return physical, logical
    logical = os.cpu_count() or 1
    return logical, logical  # can't distinguish without psutil


def _detect_system_cuda_version() -> Optional[str]:
    """Query nvidia-smi for the system CUDA driver version."""
    try:
        res = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", res.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None


def _detect_gpus_torch() -> List[GPUInfo]:
    """Detect GPUs using PyTorch CUDA."""
    active_torch = _get_torch()
    if active_torch is None or not active_torch.cuda.is_available():
        return []
    gpus: List[GPUInfo] = []
    for i in range(active_torch.cuda.device_count()):
        props = active_torch.cuda.get_device_properties(i)
        total_gb = round(props.total_memory / (1024 ** 3), 2)
        free_bytes = props.total_memory - active_torch.cuda.memory_allocated(i)
        free_gb = round(free_bytes / (1024 ** 3), 2)
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                vram_total_gb=total_gb,
                vram_free_gb=free_gb,
                cuda_available=True,
            )
        )
    return gpus


def _detect_gpus_gputil() -> List[GPUInfo]:
    """Detect GPUs using GPUtil (fallback when torch CUDA is unavailable)."""
    if not _GPUTIL:
        return []
    try:
        raw = GPUtil.getGPUs()
        return [
            GPUInfo(
                index=gpu.id,
                name=gpu.name,
                vram_total_gb=round(gpu.memoryTotal / 1024, 2),
                vram_free_gb=round(gpu.memoryFree / 1024, 2),
                cuda_available=False,  # torch CUDA not available
            )
            for gpu in raw
        ]
    except Exception:
        return []


def _detect_gpus_nvidia_smi() -> List[GPUInfo]:
    """Detect GPUs using nvidia-smi CSV output as a last resort."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode != 0:
            return []
        gpus: List[GPUInfo] = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
                name = parts[1]
                total_mb = float(parts[2])
                free_mb = float(parts[3])
                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=name,
                        vram_total_gb=round(total_mb / 1024, 2),
                        vram_free_gb=round(free_mb / 1024, 2),
                        cuda_available=True,
                    )
                )
            except ValueError:
                continue
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inspect_hardware() -> HardwareProfile:
    """Inspect the system hardware and return a :class:`HardwareProfile`.

    This function performs *only* real hardware detection — no AI, no
    guessing, no random values.  It uses the actual libraries available in
    the current environment and falls back gracefully.

    Returns
    -------
    HardwareProfile
        Immutable snapshot of the current system's hardware capabilities.
    """
    os_name = _detect_os()
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    torch_ver = _installed_torch_version()
    system_cuda = _detect_system_cuda_version()

    # NVIDIA's driver is much faster to inspect than importing PyTorch.  This
    # keeps `textlens doctor` and `textlens discover` responsive while still
    # reporting hardware directly from the installed driver.
    gpus = _detect_gpus_nvidia_smi()
    if not gpus:
        gpus = _detect_gpus_gputil()
    cuda_available = bool(gpus)
    cuda_version: Optional[str] = None

    primary_gpu = gpus[0] if gpus else None
    primary_gpu_name = primary_gpu.name if primary_gpu else None
    primary_vram = primary_gpu.vram_total_gb if primary_gpu else 0.0

    cpu_name = _detect_cpu_name()
    phys_cores, logical_cores = _detect_cpu_cores()
    ram_gb = _detect_ram_gb()

    device_type = "cuda" if cuda_available else "cpu"

    return HardwareProfile(
        os_name=os_name,
        python_version=python_ver,
        torch_version=torch_ver,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        system_cuda_version=system_cuda,
        gpus=gpus,
        primary_gpu_name=primary_gpu_name,
        primary_vram_gb=primary_vram,
        cpu_name=cpu_name,
        cpu_physical_cores=phys_cores,
        cpu_logical_cores=logical_cores,
        ram_total_gb=ram_gb,
        device_type=device_type,
    )
