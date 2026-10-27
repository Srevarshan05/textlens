"""Peak RAM / VRAM sampling for benchmarks and profiling."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Optional


def _rss_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{os.getpid()}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1e3
        except OSError:
            return None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 1e6
        except Exception:
            return None
    return None


def _gpu_used_mb() -> Optional[float]:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e6
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        for line in out.stdout.splitlines():
            pid, _, mem = line.partition(",")
            if pid.strip() == str(os.getpid()):
                return float(mem.strip())
    except Exception:
        return None
    return None


def _gpu_backend_active() -> bool:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return True
        except Exception:
            pass
    try:
        from textlens.backends.loader import get_pool

        return any(str(b.device or "").startswith(("cuda", "tensorrt")) for b in get_pool().loaded())
    except Exception:
        return False


class ResourceSampler:
    """Background sampler of peak RSS and this process's GPU memory."""

    def __init__(self, interval: float = 0.05, gpu: bool = True) -> None:
        self.interval = interval
        self.gpu = gpu
        self.peak_rss_mb: Optional[float] = None
        self.peak_vram_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_every = max(1, int(0.5 / interval))  # nvidia-smi is slow: sample ~2 Hz

    def _loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            rss = _rss_mb()
            if rss is not None:
                self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss)
            if self.gpu and n % self._gpu_every == 0:
                v = _gpu_used_mb()
                if v is not None:
                    self.peak_vram_mb = max(self.peak_vram_mb or 0.0, v)
            n += 1
            time.sleep(self.interval)

    def __enter__(self) -> "ResourceSampler":
        # Polling nvidia-smi spawns a process every ~0.5 s; only do it when a
        # GPU backend is actually in use, so CPU timings stay undisturbed.
        if self.gpu:
            self.gpu = _gpu_backend_active()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, daemon=True, name="textlens-sampler")
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    self.peak_vram_mb = max(self.peak_vram_mb or 0.0, torch.cuda.max_memory_allocated() / 1e6)
            except Exception:
                pass
