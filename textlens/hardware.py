"""
textlens.hardware
─────────────────
Hardware introspection, system NVIDIA CUDA version detection via nvidia-smi,
VRAM tracking, device status reporting, and automatic PyTorch CUDA installation
command generation for TextLens.
"""

from __future__ import annotations

import os
import re
import sys
import logging
import subprocess
import dataclasses
from typing import Optional, Dict, Any

logger = logging.getLogger("textlens.hardware")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclasses.dataclass
class SystemCUDADetails:
    """Dataclass holding system NVIDIA GPU and CUDA driver details."""
    has_nvidia_gpu: bool
    system_cuda_version: Optional[str] = None
    nvidia_smi_path: Optional[str] = None
    gpu_names: list[str] = dataclasses.field(default_factory=list)
    recommended_install_command: str = "pip install torch"


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
    system_cuda_version: Optional[str] = None
    recommended_torch_cmd: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of hardware status."""
        return dataclasses.asdict(self)


def detect_system_cuda() -> SystemCUDADetails:
    """
    Detect system NVIDIA GPU driver and installed CUDA version by querying nvidia-smi.

    Returns
    -------
    SystemCUDADetails
        Information on GPU presence, driver CUDA version, and recommended PyTorch CUDA command.
    """
    has_gpu = False
    cuda_ver = None
    gpu_list = []
    smi_path = None

    try:
        res = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if res.returncode == 0:
            has_gpu = True
            smi_out = res.stdout

            # Match CUDA Version from nvidia-smi output (e.g., CUDA Version: 13.1 or 12.4)
            match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", smi_out)
            if match:
                cuda_ver = match.group(1)

            # Match GPU Model Name(s)
            names = re.findall(r"(NVIDIA [A-Za-z0-9\s\-]+(?=\s+On|\s+Off|\s+ERR))", smi_out)
            if names:
                gpu_list = [n.strip() for n in names]
            else:
                gpu_list = ["NVIDIA GPU"]

    except Exception:
        win_smi = r"C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_\nvidia-smi.exe"
        if os.path.exists(win_smi):
            smi_path = win_smi

    recommended_cmd = get_pytorch_cuda_install_cmd(cuda_ver)

    return SystemCUDADetails(
        has_nvidia_gpu=has_gpu,
        system_cuda_version=cuda_ver,
        nvidia_smi_path=smi_path,
        gpu_names=gpu_list,
        recommended_install_command=recommended_cmd
    )


def get_pytorch_cuda_install_cmd(cuda_version: Optional[str]) -> str:
    """
    Generate the exact PyTorch pip installation command tailored to system CUDA version.

    Parameters
    ----------
    cuda_version : str, optional
        System CUDA version string (e.g., '13.1', '12.4', '12.1', '11.8').

    Returns
    -------
    str
        Pip command line for PyTorch CUDA build.
    """
    if not cuda_version:
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"

    try:
        major, minor = map(int, cuda_version.split(".")[:2])
    except ValueError:
        major, minor = 12, 4

    if major >= 13:
        # CUDA 13.x (cu130 index or cu124 compatibility)
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
    elif major == 12 and minor >= 4:
        # CUDA 12.4+
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    elif major == 12 and minor in (1, 2, 3):
        # CUDA 12.1
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    elif major == 11 and minor >= 8:
        # CUDA 11.8
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    else:
        return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"


def is_cuda_available() -> bool:
    """
    Check if NVIDIA CUDA GPU acceleration is available in active PyTorch environment.

    Returns
    -------
    bool
        True if PyTorch is compiled with CUDA and at least one CUDA GPU is accessible.
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_hardware_info() -> HardwareInfo:
    """
    Detect system GPU (CUDA) and CPU capabilities using PyTorch and nvidia-smi.

    Returns
    -------
    HardwareInfo
        Structured object containing GPU availability, GPU name, VRAM, and system details.
    """
    cpu_count = os.cpu_count() or 1
    sys_cuda = detect_system_cuda()

    if not TORCH_AVAILABLE:
        return HardwareInfo(
            gpu_available=False,
            device_type="cpu",
            gpu_count=0,
            torch_version="Not Installed",
            cpu_count=cpu_count,
            system_cuda_version=sys_cuda.system_cuda_version,
            recommended_torch_cmd=sys_cuda.recommended_install_command
        )

    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if hasattr(torch.version, 'cuda') else None
    gpu_avail = is_cuda_available()

    if gpu_avail:
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total = round(props.total_memory / (1024 ** 3), 2)
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
            cpu_count=cpu_count,
            system_cuda_version=sys_cuda.system_cuda_version,
            recommended_torch_cmd=sys_cuda.recommended_install_command
        )

    return HardwareInfo(
        gpu_available=False,
        device_type="cpu",
        gpu_count=0,
        torch_version=torch_ver,
        cuda_version=cuda_ver,
        cpu_count=cpu_count,
        system_cuda_version=sys_cuda.system_cuda_version,
        recommended_torch_cmd=sys_cuda.recommended_install_command
    )


def print_hardware_status() -> None:
    """Prints a clear, developer-friendly hardware status and CUDA setup recommendations."""
    info = get_hardware_info()
    sys_cuda = detect_system_cuda()

    print("=" * 65)
    print("           TEXTLENS HARDWARE & DEVICE STATUS           ")
    print("=" * 65)
    print(f" PyTorch Version    : {info.torch_version}")
    print(f" PyTorch CUDA Build : {info.cuda_version or 'None (CPU Only Build)'}")
    print(f" System CUDA Driver : {sys_cuda.system_cuda_version or 'Not Detected / No NVIDIA Driver'}")
    print(f" CPU Core Count     : {info.cpu_count}")

    if info.gpu_available:
        print(" CUDA (GPU) Status  : [OK] FULLY ACCELERATED")
        print(f" Active GPU Model   : {info.gpu_name}")
        print(f" Total VRAM         : {info.vram_total_gb} GB")
        print(f" VRAM Allocated     : {info.vram_allocated_gb} GB")
    elif sys_cuda.has_nvidia_gpu:
        print(" CUDA (GPU) Status  : [WARNING] NVIDIA GPU Detected, but PyTorch CUDA is missing!")
        print(f" Detected GPU       : {', '.join(sys_cuda.gpu_names) if sys_cuda.gpu_names else 'NVIDIA GPU'}")
        print(f" System CUDA Ver    : {sys_cuda.system_cuda_version or 'Unknown'}")
        print("\n [RECOMMENDED FIX TO ENABLE GPU ACCELERATION]:")
        print(" Run this exact command in your terminal to enable GPU support for your PC:")
        print(f"   {sys_cuda.recommended_install_command}")
    else:
        print(" CUDA (GPU) Status  : [NOTE] Running on CPU (No NVIDIA GPU detected)")
        print(" Active Target      : CPU (Host Processor)")
        print(" NOTE               : Multi-page document processing on CPU will be slower")
        print("                      `than CUDA GPU acceleration.")

    print("=" * 65 + "\n")
