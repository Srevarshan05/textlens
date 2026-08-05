"""
textlens.hardware
─────────────────
Hardware introspection, GPU (PyTorch CUDA) / CPU auto-detection,
VRAM tracking, and status reporting for TextLens.
"""

from __future__ import annotations
import os
import sys
import dataclasses
from typing import Optional, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclasses.dataclass
class HardwareInfo:
    """Dataclass holding system hardware capabilities for TextLens OCR."""
    gpu_available: bool
    device_type: str            # 'cuda' or 'cpu'
    gpu_count: int
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_allocated_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None
    torch_version: str = "Not Installed"
    cuda_version: Optional[str] = None
    cpu_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of hardware status."""
        return dataclasses.asdict(self)


def get_hardware_info() -> HardwareInfo:
    """
    Detect system GPU (CUDA) and CPU capabilities using PyTorch.

    Returns
    -------
    HardwareInfo
        Structured object containing GPU availability, name, VRAM, and system details.
    """
    cpu_count = os.cpu_count() or 1

    if not TORCH_AVAILABLE:
        return HardwareInfo(
            gpu_available=False,
            device_type="cpu",
            gpu_count=0,
            torch_version="Not Installed",
            cpu_count=cpu_count
        )

    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if hasattr(torch.version, 'cuda') else None
    gpu_avail = torch.cuda.is_available()

    if gpu_avail:
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total = round(props.total_memory / (1024 ** 3), 2)
        
        # Memory metrics
        vram_alloc = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        vram_res = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)

        return HardwareInfo(
            gpu_available=True,
            device_type="cuda",
            gpu_count=device_count,
            gpu_name=gpu_name,
            vram_total_gb=vram_total,
            vram_allocated_gb=vram_alloc,
            vram_reserved_gb=vram_res,
            torch_version=torch_ver,
            cuda_version=cuda_ver,
            cpu_count=cpu_count
        )

    return HardwareInfo(
        gpu_available=False,
        device_type="cpu",
        gpu_count=0,
        torch_version=torch_ver,
        cuda_version=cuda_ver,
        cpu_count=cpu_count
    )


def print_hardware_status() -> None:
    """Prints a clear, developer-friendly hardware status summary to stdout."""
    info = get_hardware_info()

    print("=" * 60)
    print("           TEXTLENS HARDWARE & DEVICE STATUS           ")
    print("=" * 60)
    print(f" PyTorch Version : {info.torch_version}")
    print(f" CUDA Compiled   : {info.cuda_version or 'N/A'}")
    print(f" CPU Cores       : {info.cpu_count}")

    if info.gpu_available:
        print(" CUDA (GPU)      : ✅ AVAILABLE (100% Accelerated)")
        print(f" GPU Model       : {info.gpu_name}")
        print(f" Total VRAM      : {info.vram_total_gb} GB")
        print(f" VRAM Allocated  : {info.vram_allocated_gb} GB")
        print(f" Active Target   : CUDA (Default model target: 'cuda')")
    else:
        print(" CUDA (GPU)      : ⚠️  NOT DETECTED (Running on CPU)")
        print(" Active Target   : CPU (Model will execute on host processor)")

    print("=" * 60 + "\n")
