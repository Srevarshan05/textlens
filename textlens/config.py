"""
textlens.config
───────────────
Runtime configuration for TextLens.

Every setting has a sensible default and can be overridden by an environment
variable (useful for containers) or by the user config file written by
``textlens setup``.  Precedence, highest first::

    explicit argument  >  environment variable  >  config file  >  default

Nothing in this module touches the filesystem at import time.  Directories
are created lazily by the component that writes to them.

Environment variables
---------------------
=============================  ==============================================
``TEXTLENS_HOME``              Root for models, caches and config
                               (default ``~/.cache/textlens``)
``TEXTLENS_MODELS_DIR``        Model weights (default ``$TEXTLENS_HOME/models``)
``TEXTLENS_CACHE_DIR``         Result cache (default ``$TEXTLENS_HOME/results``)
``TEXTLENS_PROFILE``           Default routing profile (default ``auto``)
``TEXTLENS_DEVICE``            Force a device: ``cpu``, ``cuda``, ``cuda:1``…
``TEXTLENS_OFFLINE``           ``1`` = never download; use local models only
``TEXTLENS_NO_PERSIST``        ``1`` = never write documents/results to disk
``TEXTLENS_RESULT_CACHE``      ``off`` | ``memory`` (default) | ``disk``
``TEXTLENS_LOG_LEVEL``         ``DEBUG`` … ``ERROR`` (default ``WARNING``)
``TEXTLENS_LOG_FORMAT``        ``text`` (default) or ``json``
``TEXTLENS_MAX_PAGES``         Refuse documents with more pages (default 5000)
``TEXTLENS_MAX_FILE_MB``       Refuse larger inputs (default 512)
``TEXTLENS_ALLOW_URLS``        ``1`` = allow http(s) URL inputs (default ``1``
                               for the library, the server has its own flag)
=============================  ==============================================

Server-only settings live in :class:`textlens.serving.settings.ServerSettings`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger("textlens.config")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

PROFILES = ("auto", "edge", "fast", "balanced", "accurate", "document", "server")
RESULT_CACHE_MODES = ("off", "memory", "disk")


def _default_home() -> Path:
    # ``~/.cache/textlens`` is the historical location on every platform;
    # keeping it means previously downloaded models remain installed.
    return Path.home() / ".cache" / "textlens"


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning("Ignoring %s=%r (expected a boolean)", key, raw)
    return default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r (expected an integer)", key, raw)
        return default


@dataclasses.dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the effective TextLens configuration."""

    home: Path
    models_dir: Path
    results_dir: Path
    profile: str = "auto"
    device: Optional[str] = None
    offline: bool = False
    no_persist: bool = False
    result_cache: str = "memory"
    log_level: str = "WARNING"
    log_format: str = "text"
    max_pages: int = 5000
    max_file_mb: int = 512
    allow_urls: bool = True
    hf_token: Optional[str] = None

    @property
    def config_file(self) -> Path:
        return self.home / "config.json"

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        file_values: Optional[Mapping[str, Any]] = None,
    ) -> "Settings":
        env = os.environ if env is None else env
        home = Path(env.get("TEXTLENS_HOME") or _default_home()).expanduser()
        if file_values is None:
            file_values = _read_config_file(home / "config.json")
        fv = dict(file_values or {})

        def pick(key: str, env_key: str, default: Any) -> Any:
            if env.get(env_key):
                return env[env_key]
            return fv.get(key, default)

        profile = str(pick("profile", "TEXTLENS_PROFILE", "auto")).lower()
        if profile not in PROFILES:
            logger.warning("Unknown profile %r; falling back to 'auto'", profile)
            profile = "auto"
        cache_mode = str(pick("result_cache", "TEXTLENS_RESULT_CACHE", "memory")).lower()
        if cache_mode not in RESULT_CACHE_MODES:
            logger.warning("Unknown result cache mode %r; using 'memory'", cache_mode)
            cache_mode = "memory"
        no_persist = _env_bool(env, "TEXTLENS_NO_PERSIST", bool(fv.get("no_persist", False)))
        if no_persist and cache_mode == "disk":
            cache_mode = "memory"
        offline = _env_bool(
            env,
            "TEXTLENS_OFFLINE",
            _env_bool(env, "HF_HUB_OFFLINE", bool(fv.get("offline", False))),
        )
        return cls(
            home=home,
            models_dir=Path(env.get("TEXTLENS_MODELS_DIR") or fv.get("models_dir") or home / "models").expanduser(),
            results_dir=Path(env.get("TEXTLENS_CACHE_DIR") or fv.get("results_dir") or home / "results").expanduser(),
            profile=profile,
            device=pick("device", "TEXTLENS_DEVICE", None) or None,
            offline=offline,
            no_persist=no_persist,
            result_cache=cache_mode,
            log_level=str(pick("log_level", "TEXTLENS_LOG_LEVEL", "WARNING")).upper(),
            log_format=str(pick("log_format", "TEXTLENS_LOG_FORMAT", "text")).lower(),
            max_pages=_env_int(env, "TEXTLENS_MAX_PAGES", int(fv.get("max_pages", 5000))),
            max_file_mb=_env_int(env, "TEXTLENS_MAX_FILE_MB", int(fv.get("max_file_mb", 512))),
            allow_urls=_env_bool(env, "TEXTLENS_ALLOW_URLS", bool(fv.get("allow_urls", True))),
            hf_token=env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["hf_token"] = "***" if self.hf_token else None
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}


def _read_config_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable config file %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def write_config_file(values: Mapping[str, Any], settings: Optional[Settings] = None) -> Path:
    """Merge *values* into the user config file and return its path."""
    settings = settings or get_settings()
    path = settings.config_file
    current = _read_config_file(path)
    current.update({k: v for k, v in values.items() if v is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    reload_settings()
    return path


_lock = threading.Lock()
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide settings snapshot (computed on first use)."""
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """Re-read environment and config file (used by tests and ``setup``)."""
    global _settings
    with _lock:
        _settings = Settings.from_env()
    return _settings
