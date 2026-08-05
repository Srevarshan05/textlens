"""
textlens.models.manager
────────────────────────
ModelManager — the primary developer-facing interface for managing TextLens models.

Responsibilities
----------------
- List all officially supported models with their install status.
- Download / cache individual models from HuggingFace.
- Remove cached models from disk.
- Show detailed info for a specific model.

None of these methods reach out to arbitrary HuggingFace repositories.
They operate exclusively on models registered in the central registry.

Usage
-----
    from textlens.models import ModelManager

    ModelManager.models()
    ModelManager.download("glm-ocr")
    ModelManager.info("florence2")
    ModelManager.remove("smolvlm")
"""

from __future__ import annotations

import logging
from typing import List, Optional

from textlens.models.cache import ModelCache
from textlens.models.downloader import ModelDownloader
from textlens.models.exceptions import UnknownModelError
from textlens.models.metadata import ModelMetadata
from textlens.models.registry import ModelRegistry

logger = logging.getLogger("textlens.models.manager")


# ---------------------------------------------------------------------------
# Shared singletons (module-level, lazily initialised)
# ---------------------------------------------------------------------------

_cache = ModelCache()
_downloader = ModelDownloader(cache=_cache)


def _get_console():  # type: ignore[return]
    try:
        from rich.console import Console
        return Console(force_terminal=True, highlight=False)
    except ImportError:
        return None


class ModelManager:
    """Developer-facing class for managing the TextLens model catalog.

    All methods are class-methods; no instantiation is required.

    Example
    -------
    ::

        >>> ModelManager.models()
        >>> ModelManager.download("glm-ocr")
        >>> ModelManager.info("florence2")
        >>> ModelManager.remove("smolvlm")
    """

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    @classmethod
    def models(cls) -> List[ModelMetadata]:
        """Print and return all officially supported TextLens models.

        Output includes name, parameter count, key use-cases, hardware
        requirements, and whether the model is currently installed.

        Returns
        -------
        list[ModelMetadata]
            All registered models in catalog order.
        """
        catalog = ModelRegistry.all()
        console = _get_console()

        if console:
            cls._print_models_rich(catalog, console)
        else:
            cls._print_models_plain(catalog)

        return catalog

    @classmethod
    def _print_models_rich(cls, catalog: List[ModelMetadata], console) -> None:
        try:
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich import box

            table = Table(
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 2),
                expand=True,
            )
            table.add_column("Model ID", min_width=14, style="bold yellow")
            table.add_column("Official HF Repo", min_width=24, style="dim cyan")
            table.add_column("Params", min_width=8, justify="right")
            table.add_column("Use Cases", min_width=30)
            table.add_column("Hardware", min_width=12, style="dim")
            table.add_column("Installed", min_width=10, justify="center")

            tag_colors = ["cyan", "green", "magenta", "yellow", "bright_cyan", "bright_green"]

            for meta in catalog:
                installed = _cache.is_installed(meta.id)
                installed_text = Text("✓ Yes", style="bold green") if installed else Text("✗ No", style="dim red")
                default_tag = " [Default]" if meta.is_default else ""

                colored_tags = [
                    f"[{tag_colors[i % len(tag_colors)]}]{tag}[/{tag_colors[i % len(tag_colors)]}]"
                    for i, tag in enumerate(meta.use_cases[:2])
                ]
                use_cases_formatted = ", ".join(colored_tags)

                table.add_row(
                    f"{meta.id}{default_tag}",
                    meta.hf_repo_id,
                    meta.parameters,
                    use_cases_formatted,
                    meta.min_recommendation,
                    installed_text,
                )

            console.print()
            console.print(
                Panel(
                    table,
                    title="[bold]TextLens — Official Supported Models Catalog[/bold]",
                    border_style="cyan",
                )
            )
            console.print(
                "[dim]Use [bold]OCR(model='<Model ID>')[/bold] or [bold]textlens model install <Model ID>[/bold] to use a model.[/dim]\n"
            )
        except ImportError:
            cls._print_models_plain(catalog)

    @classmethod
    def _print_models_plain(cls, catalog: List[ModelMetadata]) -> None:
        sep = "-" * 52
        print(f"\n{sep}")
        print(" TextLens — Supported Models")
        print(sep)
        for meta in catalog:
            installed = _cache.is_installed(meta.id)
            inst_label = "Installed: Yes" if installed else "Installed: No"
            default_tag = " [Default]" if meta.is_default else ""
            print(f"\n{meta.display_name}{default_tag}")
            print(f"  ID        : {meta.id}")
            print(f"  Category  : {meta.category}")
            print(f"  Params    : {meta.parameters}")
            print(f"  Use Cases : {', '.join(meta.use_cases[:2])}")
            print(f"  Hardware  : {meta.min_recommendation}")
            print(f"  {inst_label}")
            print(sep)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @classmethod
    def download(cls, model_id: str) -> None:
        """Download and cache a TextLens model from HuggingFace Hub.

        Parameters
        ----------
        model_id : str
            Canonical model slug (e.g. ``"glm-ocr"``).

        Raises
        ------
        textlens.models.exceptions.UnknownModelError
            If *model_id* is not in the official catalog.
        textlens.models.exceptions.DownloadError
            If the download fails for any reason.

        Notes
        -----
        - If the model is already installed this method prints a confirmation
          and returns immediately without re-downloading.
        - Only TextLens-registered models can be downloaded.  Arbitrary
          HuggingFace repository IDs are not accepted.
        """
        _downloader.download(model_id)

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    @classmethod
    def remove(cls, model_id: str) -> None:
        """Remove a cached model from disk.

        Parameters
        ----------
        model_id : str
            Canonical model slug.

        Raises
        ------
        textlens.models.exceptions.UnknownModelError
            If *model_id* is not in the official catalog.
        """
        _downloader.remove(model_id)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @classmethod
    def info(cls, model_id: str) -> ModelMetadata:
        """Print detailed information about a specific model and return its metadata.

        Printed information includes:
        - Description
        - Parameter count
        - Category / use-cases
        - Hardware recommendation
        - Installation status + disk usage
        - Approximate download size

        Parameters
        ----------
        model_id : str
            Canonical model slug (e.g. ``"glm-ocr"``).

        Returns
        -------
        ModelMetadata
            The metadata for the requested model.

        Raises
        ------
        textlens.models.exceptions.UnknownModelError
            If *model_id* is not in the official catalog.
        """
        meta = ModelRegistry.get(model_id)
        installed = _cache.is_installed(model_id)
        disk_gb = _cache.disk_usage_gb(model_id)
        cache_path = _cache.model_path(model_id)
        console = _get_console()

        if console:
            cls._print_info_rich(meta, installed, disk_gb, cache_path, console)
        else:
            cls._print_info_plain(meta, installed, disk_gb, cache_path)

        return meta

    @classmethod
    def _print_info_rich(cls, meta, installed, disk_gb, cache_path, console) -> None:
        try:
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich import box

            table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            table.add_column("Field", style="dim", min_width=20)
            table.add_column("Value", style="bold white")

            table.add_row("ID", meta.id)
            table.add_row("Category", meta.category)
            table.add_row("Parameters", meta.parameters)
            table.add_row("HuggingFace Repo", meta.hf_repo_id)
            table.add_row("Use Cases", ", ".join(meta.use_cases))
            table.add_row("Min Hardware", meta.min_recommendation)
            table.add_row("CPU Supported", "Yes" if meta.cpu_supported else "No")
            table.add_row("Default Model", "Yes" if meta.is_default else "No")

            install_val = (
                Text(f"Yes ({disk_gb:.2f} GB)", style="bold green")
                if installed
                else Text("No", style="dim red")
            )
            table.add_row("Installed", install_val)

            dl_size = f"~{meta.download_size_gb} GB" if meta.download_size_gb else "Unknown"
            table.add_row("Download Size", dl_size)

            if installed:
                table.add_row("Cache Path", str(cache_path))

            desc_panel = Panel(
                meta.description,
                title="Description",
                border_style="dim",
                padding=(0, 2),
            )

            console.print()
            console.print(
                Panel(
                    table,
                    title=f"[bold]{meta.display_name}[/bold]",
                    border_style="cyan",
                )
            )
            console.print(desc_panel)
            console.print()
        except ImportError:
            cls._print_info_plain(meta, installed, disk_gb, cache_path)

    @classmethod
    def _print_info_plain(cls, meta, installed, disk_gb, cache_path) -> None:
        sep = "-" * 52
        print(f"\n{sep}")
        print(f" {meta.display_name}")
        print(sep)
        print(f"  ID          : {meta.id}")
        print(f"  Category    : {meta.category}")
        print(f"  Parameters  : {meta.parameters}")
        print(f"  HF Repo     : {meta.hf_repo_id}")
        print(f"  Use Cases   : {', '.join(meta.use_cases)}")
        print(f"  Hardware    : {meta.min_recommendation}")
        print(f"  CPU Support : {'Yes' if meta.cpu_supported else 'No'}")
        print(f"  Default     : {'Yes' if meta.is_default else 'No'}")
        if installed:
            print(f"  Installed   : Yes ({disk_gb:.2f} GB)")
            print(f"  Cache Path  : {cache_path}")
        else:
            print(f"  Installed   : No")
        dl_size = f"~{meta.download_size_gb} GB" if meta.download_size_gb else "Unknown"
        print(f"  DL Size     : {dl_size}")
        print(f"\n  {meta.description}")
        print(sep)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @classmethod
    def is_installed(cls, model_id: str) -> bool:
        """Return ``True`` if *model_id* is cached locally.

        Raises
        ------
        textlens.models.exceptions.UnknownModelError
            If *model_id* is not in the official catalog.
        """
        ModelRegistry.get(model_id)  # validates the ID
        return _cache.is_installed(model_id)
