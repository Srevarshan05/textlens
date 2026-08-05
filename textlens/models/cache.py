"""
textlens.models.cache
──────────────────────
Local model cache manager.

TextLens stores all downloaded model weights in::

    ~/.cache/textlens/models/<model-id>/

This module provides a single ``ModelCache`` class that abstracts all
filesystem operations so the rest of the codebase never constructs raw paths.

Usage
-----
    from textlens.models.cache import ModelCache

    cache = ModelCache()
    path = cache.model_path("glm-ocr")      # pathlib.Path
    installed = cache.is_installed("glm-ocr")
    size_gb = cache.disk_usage_gb("glm-ocr")
    cache.remove("glm-ocr")
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("textlens.models.cache")

# Root cache directory: ~/.cache/textlens/models/
_CACHE_ROOT = Path.home() / ".cache" / "textlens" / "models"


class ModelCache:
    """Manages local disk cache for TextLens model weights.

    Attributes
    ----------
    root : pathlib.Path
        The root directory where all model subdirectories live.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root: Path = root or _CACHE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def model_path(self, model_id: str) -> Path:
        """Return the local directory path for a given model.

        The directory may or may not exist yet.  Use :meth:`is_installed`
        to check before attempting to load files.

        Parameters
        ----------
        model_id : str
            Canonical model slug (e.g. ``"glm-ocr"``).

        Returns
        -------
        pathlib.Path
            Absolute path to the model's local cache directory.
        """
        return self.root / model_id

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def is_installed(self, model_id: str) -> bool:
        """Return ``True`` if the model directory exists and is non-empty.

        An empty directory is treated as *not installed* because a partial
        or interrupted download leaves an empty folder.
        """
        path = self.model_path(model_id)
        if not path.exists() or not path.is_dir():
            return False
        # Check the directory contains at least one file
        return any(path.iterdir())

    def disk_usage_gb(self, model_id: str) -> float:
        """Return the total disk usage of a cached model in gigabytes.

        Returns ``0.0`` if the model is not installed.
        """
        path = self.model_path(model_id)
        if not path.exists():
            return 0.0

        total_bytes = sum(
            f.stat().st_size
            for f in path.rglob("*")
            if f.is_file()
        )
        return round(total_bytes / (1024 ** 3), 3)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def remove(self, model_id: str) -> bool:
        """Delete a cached model from disk.

        Parameters
        ----------
        model_id : str
            Canonical model slug.

        Returns
        -------
        bool
            ``True`` if the directory existed and was deleted, ``False``
            if the model was not cached.
        """
        path = self.model_path(model_id)
        if path.exists():
            shutil.rmtree(path)
            logger.info("Removed cached model: %s (%s)", model_id, path)
            return True
        logger.warning("Cannot remove '%s': not found in cache.", model_id)
        return False

    def ensure_directory(self, model_id: str) -> Path:
        """Ensure the model's cache directory exists and return its path."""
        path = self.model_path(model_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_installed(self) -> list[str]:
        """Return a list of model IDs that are currently cached on disk."""
        if not self.root.exists():
            return []
        return [
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and any(d.iterdir())
        ]

    def __repr__(self) -> str:
        return f"ModelCache(root={self.root!r})"
