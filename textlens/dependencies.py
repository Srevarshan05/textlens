"""
textlens.dependencies
────────────────────
Automated dependency checker and environment self-healer for TextLens.
Ensures required packages (torch, transformers, accelerate, pillow, pypdfium2, fastapi, etc.)
are properly installed and offers automatic repair if any package is missing.
"""

from __future__ import annotations

import sys
import subprocess
import importlib
import dataclasses
from typing import List, Dict, Tuple, Optional


REQUIRED_PACKAGES = [
    ("torch", "torch>=2.0.0"),
    ("transformers", "transformers>=4.40.0"),
    ("accelerate", "accelerate>=0.28.0"),
    ("PIL", "pillow>=9.0.0"),
    ("pypdfium2", "pypdfium2>=4.20.0"),
    ("pydantic", "pydantic>=2.0.0"),
    ("fastapi", "fastapi>=0.100.0"),
    ("uvicorn", "uvicorn>=0.22.0"),
    ("multipart", "python-multipart>=0.0.6"),
]


@dataclasses.dataclass
class DependencyReport:
    all_satisfied: bool
    installed: List[str]
    missing: List[str]
    install_command: str


def check_dependencies() -> DependencyReport:
    """
    Check if all required dependencies for TextLens are installed in current Python environment.

    Returns
    -------
    DependencyReport
        Object detailing satisfied packages, missing packages, and install commands.
    """
    installed = []
    missing = []
    missing_pip = []

    for mod_name, pip_spec in REQUIRED_PACKAGES:
        try:
            importlib.import_module(mod_name)
            installed.append(pip_spec.split(">=")[0])
        except ImportError:
            missing.append(mod_name)
            missing_pip.append(pip_spec)

    all_satisfied = len(missing) == 0
    install_cmd = f"pip install {' '.join(missing_pip)}" if missing_pip else ""

    return DependencyReport(
        all_satisfied=all_satisfied,
        installed=installed,
        missing=missing,
        install_command=install_cmd
    )


def ensure_dependencies(auto_install: bool = False, verbose: bool = True) -> bool:
    """
    Validate environment dependencies and optionally auto-install missing packages.

    Parameters
    ----------
    auto_install : bool
        If True, automatically runs `pip install` for missing packages.
    verbose : bool
        If True, prints status messages to stdout.

    Returns
    -------
    bool
        True if all dependencies are satisfied after check/install.
    """
    report = check_dependencies()

    if report.all_satisfied:
        if verbose:
            print("[TextLens Environment] [OK] All required dependencies are installed.")
        return True

    if verbose:
        print("[TextLens Environment] [WARNING] Missing dependencies detected:")
        for pkg in report.missing:
            print(f"  - {pkg}")

    if auto_install:
        if verbose:
            print(f"[TextLens Auto-Repair] Installing missing packages: {report.install_command} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + report.install_command.split()[2:])
            if verbose:
                print("[TextLens Auto-Repair] [OK] Missing dependencies successfully installed!")
            return True
        except Exception as err:
            if verbose:
                print(f"[TextLens Auto-Repair] [ERROR] Auto-installation failed: {err}")
                print(f"Please run manually: {report.install_command}")
            return False
    else:
        if verbose:
            print(f"\nTo fix missing packages, run this command:\n  {report.install_command}\n")
        return False
