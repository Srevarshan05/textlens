"""
textlens.models.artifacts
─────────────────────────
Integrity-checked model artifact installation.

Two sources are supported:

* **Pinned URL artifacts** (e.g. PP-OCRv6 ONNX files): streamed to a
  temporary file, verified against the SHA-256 in the model spec, then
  atomically renamed into place.  A corrupted or tampered file is never
  installed.
* **Hugging Face repositories** (document VLMs): downloaded with
  ``huggingface_hub.snapshot_download``, pinned to ``spec.revision`` when
  the spec provides one.

Offline mode (``TEXTLENS_OFFLINE=1``) turns every would-be download into an
:class:`~textlens.errors.OfflineError` with instructions.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from textlens.errors import DownloadError, IntegrityError, OfflineError
from textlens.models.specs import Artifact, ModelSpec

logger = logging.getLogger("textlens.models.artifacts")

ProgressFn = Callable[[str, int, Optional[int]], None]  # (filename, done, total)


def model_dir(spec: ModelSpec, root: Optional[Path] = None) -> Path:
    from textlens.config import get_settings

    return Path(root or get_settings().models_dir) / spec.id


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifacts_present(spec: ModelSpec, root: Optional[Path] = None, verify: bool = False) -> bool:
    d = model_dir(spec, root)
    for art in spec.artifacts:
        p = d / art.filename
        if not p.is_file():
            return False
        if art.size_bytes is not None and p.stat().st_size != art.size_bytes:
            return False
        if verify and sha256_file(p) != art.sha256:
            return False
    return True


def is_installed(spec: ModelSpec, root: Optional[Path] = None) -> bool:
    if spec.artifacts:
        return artifacts_present(spec, root)
    d = model_dir(spec, root)
    return d.is_dir() and any(d.iterdir())


def _download_one(art: Artifact, dest: Path, progress: Optional[ProgressFn], timeout: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{art.filename}.", suffix=".part", dir=str(dest.parent))
    tmp = Path(tmp_name)
    h = hashlib.sha256()
    try:
        req = urllib.request.Request(art.url, headers={"User-Agent": "textlens"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, os.fdopen(fd, "wb") as out:  # noqa: S310
            total = resp.headers.get("Content-Length")
            total_n = int(total) if total and total.isdigit() else art.size_bytes
            done = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(art.filename, done, total_n)
        digest = h.hexdigest()
        if digest != art.sha256:
            raise IntegrityError(art.filename, f"SHA-256 mismatch (expected {art.sha256[:12]}…, got {digest[:12]}…)")
        os.replace(tmp, dest)
    except IntegrityError:
        raise
    except Exception as exc:
        raise DownloadError(art.filename, f"{exc} ({art.url})") from exc
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def install(
    spec: ModelSpec,
    force: bool = False,
    root: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
    timeout: float = 300.0,
    hf_token: Optional[str] = None,
) -> Path:
    """Install *spec*'s weights into the model cache and return the folder."""
    from textlens.config import get_settings

    settings = get_settings()
    target = model_dir(spec, root)
    if not force and is_installed(spec, root):
        return target
    if settings.offline:
        raise OfflineError(
            f"Model {spec.id!r} is not installed and offline mode is enabled.",
            hint=f"Copy the model into {target} or unset TEXTLENS_OFFLINE and run `textlens models install {spec.id}`.",
        )
    if settings.no_persist:
        logger.info("no-persist mode only restricts documents/results; model weights are still cached.")

    t0 = time.perf_counter()
    if spec.artifacts:
        for art in spec.artifacts:
            dest = target / art.filename
            if not force and dest.is_file() and (art.size_bytes is None or dest.stat().st_size == art.size_bytes):
                continue
            _download_one(art, dest, progress, timeout)
    elif spec.hf_repo_id:
        try:
            import huggingface_hub
        except ImportError as exc:
            raise DownloadError(
                spec.id, 'huggingface_hub is not installed. Run: pip install "textlens-ocr[gpu]"'
            ) from exc
        target.mkdir(parents=True, exist_ok=True)
        try:
            huggingface_hub.snapshot_download(
                repo_id=spec.hf_repo_id,
                revision=spec.revision,
                local_dir=str(target),
                token=hf_token or settings.hf_token,
            )
        except Exception as exc:
            if not any(target.iterdir()):
                shutil.rmtree(target, ignore_errors=True)
            raise DownloadError(spec.id, str(exc)) from exc
    else:
        raise DownloadError(spec.id, "this catalog entry has no downloadable weights (served or external model)")
    logger.info("Installed %s in %.1fs → %s", spec.id, time.perf_counter() - t0, target)
    return target


def verify(spec: ModelSpec, root: Optional[Path] = None) -> dict:
    """Re-hash installed artifacts; returns ``{filename: ok}``."""
    d = model_dir(spec, root)
    return {a.filename: (d / a.filename).is_file() and sha256_file(d / a.filename) == a.sha256 for a in spec.artifacts}
