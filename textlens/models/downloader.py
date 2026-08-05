"""
textlens.models.downloader
───────────────────────────
HuggingFace model downloader with progress tracking and duplicate detection.

Features
--------
- Downloads via ``huggingface_hub.snapshot_download`` for full repo snapshots.
- Rich progress output — always prints status, never silent.
- Duplicate detection — checks cache before downloading.
- Friendly error messages on failure.

Usage
-----
    from textlens.models.downloader import ModelDownloader
    from textlens.models.cache import ModelCache
    from textlens.models.registry import ModelRegistry

    dl = ModelDownloader(cache=ModelCache())
    dl.download("glm-ocr")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("textlens.models.downloader")

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TransferSpeedColumn,
    )
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False

try:
    import huggingface_hub
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


from textlens.models.cache import ModelCache
from textlens.models.exceptions import DownloadError
from textlens.models.registry import ModelRegistry


def _get_console() -> "Console":  # type: ignore[name-defined]
    if _RICH:
        return Console()
    raise RuntimeError("rich is not installed")


class ModelDownloader:
    """Downloads TextLens models from HuggingFace Hub to local cache.

    Parameters
    ----------
    cache : ModelCache, optional
        Cache manager instance.  A default one is created if not provided.
    hf_token : str, optional
        HuggingFace access token for gated repositories.
    """

    def __init__(
        self,
        cache: Optional[ModelCache] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self._cache = cache or ModelCache()
        self._hf_token = hf_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, model_id: str, force: bool = False) -> Path:
        """Download a model from HuggingFace to the local cache.

        Parameters
        ----------
        model_id : str
            Canonical TextLens model slug (e.g. ``"glm-ocr"``).
        force : bool
            If ``True``, re-download even if the model is already cached.

        Returns
        -------
        pathlib.Path
            The local directory containing the downloaded model files.

        Raises
        ------
        textlens.models.exceptions.UnknownModelError
            If *model_id* is not in the TextLens model registry.
        textlens.models.exceptions.DownloadError
            If the download fails for any reason.
        """
        # Validate — raises UnknownModelError if not in registry
        meta = ModelRegistry.get(model_id)

        console = Console(force_terminal=True, highlight=False) if _RICH else None

        # ── Already installed? ──────────────────────────────────────────
        if not force and self._cache.is_installed(model_id):
            msg = f"{meta.display_name} already installed."
            if console:
                console.print(f"[bold green][OK][/bold green] {msg}")
            else:
                print(f"[OK] {msg}")
            return self._cache.model_path(model_id)

        # ── Check huggingface_hub available ─────────────────────────────
        if not _HF_AVAILABLE:
            raise DownloadError(
                model_id,
                "huggingface_hub is not installed. "
                "Run: pip install huggingface_hub",
            )

        local_dir = self._cache.ensure_directory(model_id)

        # ── Print header ────────────────────────────────────────────────
        if console:
            console.print()
            console.print(
                Panel(
                    Text.from_markup(
                        f"[bold cyan]Downloading[/bold cyan] [bold white]{meta.display_name}[/bold white]\n"
                        f"[dim]Repository:[/dim] [blue]{meta.hf_repo_id}[/blue]\n"
                        f"[dim]Cache path:[/dim] [dim]{local_dir}[/dim]"
                    ),
                    title="[bold]TextLens Model Installer[/bold]",
                    border_style="cyan",
                    padding=(0, 2),
                )
            )
        else:
            print(f"\nDownloading {meta.display_name} ({meta.hf_repo_id}) ...")
            print(f"  Cache path: {local_dir}")

        # ── Download ────────────────────────────────────────────────────
        start = time.time()
        if _HF_AVAILABLE:
            try:
                disable_progress_bars()
            except Exception:
                pass

        try:
            if console:
                with Progress(
                    SpinnerColumn("line"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,
                ) as progress:
                    task = progress.add_task(
                        f"[cyan]{meta.display_name}[/cyan]",
                        total=None,  # indeterminate — HF handles chunking
                    )
                    downloaded_path = huggingface_hub.snapshot_download(
                        repo_id=meta.hf_repo_id,
                        local_dir=str(local_dir),
                        token=self._hf_token,
                    )
                    progress.update(task, completed=100, total=100)
            else:
                downloaded_path = huggingface_hub.snapshot_download(
                    repo_id=meta.hf_repo_id,
                    local_dir=str(local_dir),
                    token=self._hf_token,
                )

        except Exception as exc:
            raise DownloadError(model_id, str(exc)) from exc
        finally:
            if _HF_AVAILABLE:
                try:
                    enable_progress_bars()
                except Exception:
                    pass

        elapsed = round(time.time() - start, 1)
        disk_gb = self._cache.disk_usage_gb(model_id)

        # ── Success summary ─────────────────────────────────────────────
        if console:
            console.print()
            console.print(
                f"[bold green][OK] {meta.display_name} installed successfully![/bold green] "
                f"[dim]({disk_gb:.2f} GB · {elapsed}s)[/dim]"
            )
            console.print(f"  [dim]Location:[/dim] {local_dir}")
        else:
            print(f"\n[OK] {meta.display_name} installed successfully!")
            print(f"  Disk usage : {disk_gb:.2f} GB")
            print(f"  Time       : {elapsed}s")
            print(f"  Location   : {local_dir}")

        logger.info(
            "Model '%s' downloaded in %.1fs (%.3f GB) → %s",
            model_id, elapsed, disk_gb, local_dir,
        )
        return Path(downloaded_path)

    def remove(self, model_id: str) -> bool:
        """Remove a cached model from disk.

        Parameters
        ----------
        model_id : str
            Canonical TextLens model slug.

        Returns
        -------
        bool
            ``True`` if the model was removed, ``False`` if it wasn't cached.
        """
        # Validate first
        meta = ModelRegistry.get(model_id)
        console = Console(force_terminal=True, highlight=False) if _RICH else None

        removed = self._cache.remove(model_id)
        if removed:
            msg = f"{meta.display_name} removed from cache."
            if console:
                console.print(f"[bold yellow][OK][/bold yellow] {msg}")
            else:
                print(f"[OK] {msg}")
        else:
            msg = f"{meta.display_name} is not installed - nothing to remove."
            if console:
                console.print(f"[dim]ℹ {msg}[/dim]")
            else:
                print(f"ℹ {msg}")

        return removed
