"""
textlens.serving.settings
─────────────────────────
Server configuration from environment variables (container-friendly).

=============================  ==========================================  =========
Variable                        Meaning                                     Default
=============================  ==========================================  =========
``TEXTLENS_HOST``               bind address                                127.0.0.1
``TEXTLENS_PORT``               port                                        8000
``TEXTLENS_PROFILE``            routing profile                             auto
``TEXTLENS_MODEL``              pin a model for every request               (routed)
``TEXTLENS_DEVICE``             cpu / cuda / cuda:N …                       auto
``TEXTLENS_API_KEYS``           comma-separated keys (auth off if empty)    —
``TEXTLENS_MAX_UPLOAD_MB``      per-request upload limit                    50
``TEXTLENS_MAX_PAGES``          pages per document                          500
``TEXTLENS_WORKERS``            concurrent inference jobs                   1
``TEXTLENS_MAX_QUEUE``          waiting jobs before 503 (backpressure)      64
``TEXTLENS_RATE_LIMIT``         requests/second per client (0 = off)        0
``TEXTLENS_RATE_BURST``         token-bucket burst size                     10
``TEXTLENS_SERVER_ALLOW_URLS``  accept http(s) URL inputs (SSRF-guarded)    0
``TEXTLENS_NO_PERSIST``         never write uploads/results to disk         0
``TEXTLENS_CORS_ORIGINS``       comma-separated allowed origins             —
``TEXTLENS_JOB_TTL``            seconds to keep finished jobs               3600
``TEXTLENS_SYNC_TIMEOUT``       max seconds a sync request waits            300
``TEXTLENS_METRICS_PUBLIC``     /metrics without an API key                 1
``TEXTLENS_WARMUP``             load the default model at startup           0
``TEXTLENS_ALLOWED_ROOT``       legacy ``image_url`` local paths root       — (off)
=============================  ==========================================  =========
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping, Optional, Tuple

_TRUE = {"1", "true", "yes", "on"}


def _b(env: Mapping[str, str], key: str, default: bool) -> bool:
    v = env.get(key)
    return default if v is None or v == "" else v.strip().lower() in _TRUE


def _i(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _f(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _list(env: Mapping[str, str], key: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in env.get(key, "").split(",") if x.strip())


@dataclasses.dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    profile: Optional[str] = None
    model: Optional[str] = None
    device: Optional[str] = None
    endpoint: Optional[str] = None
    api_keys: Tuple[str, ...] = ()
    max_upload_mb: int = 50
    max_pages: int = 500
    workers: int = 1
    max_queue: int = 64
    rate_limit: float = 0.0
    rate_burst: int = 10
    allow_urls: bool = False
    no_persist: bool = False
    cors_origins: Tuple[str, ...] = ()
    job_ttl: float = 3600.0
    sync_timeout: float = 300.0
    metrics_public: bool = True
    warmup: bool = False
    allowed_root: Optional[str] = None
    openai_api: bool = True
    batch_max_files: int = 100

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ServerSettings":
        env = os.environ if env is None else env
        return cls(
            host=env.get("TEXTLENS_HOST", "127.0.0.1"),
            port=_i(env, "TEXTLENS_PORT", 8000),
            profile=env.get("TEXTLENS_PROFILE") or None,
            model=env.get("TEXTLENS_MODEL") or None,
            device=env.get("TEXTLENS_DEVICE") or None,
            endpoint=env.get("TEXTLENS_OPENAI_BASE_URL") or None,
            api_keys=_list(env, "TEXTLENS_API_KEYS"),
            max_upload_mb=_i(env, "TEXTLENS_MAX_UPLOAD_MB", 50),
            max_pages=_i(env, "TEXTLENS_MAX_PAGES", 500),
            workers=_i(env, "TEXTLENS_WORKERS", 1),
            max_queue=_i(env, "TEXTLENS_MAX_QUEUE", 64),
            rate_limit=_f(env, "TEXTLENS_RATE_LIMIT", 0.0),
            rate_burst=_i(env, "TEXTLENS_RATE_BURST", 10),
            allow_urls=_b(env, "TEXTLENS_SERVER_ALLOW_URLS", False),
            no_persist=_b(env, "TEXTLENS_NO_PERSIST", False),
            cors_origins=_list(env, "TEXTLENS_CORS_ORIGINS"),
            job_ttl=_f(env, "TEXTLENS_JOB_TTL", 3600.0),
            sync_timeout=_f(env, "TEXTLENS_SYNC_TIMEOUT", 300.0),
            metrics_public=_b(env, "TEXTLENS_METRICS_PUBLIC", True),
            warmup=_b(env, "TEXTLENS_WARMUP", False),
            allowed_root=env.get("TEXTLENS_ALLOWED_ROOT") or None,
        )

    def replace(self, **changes: Any) -> "ServerSettings":
        return dataclasses.replace(self, **changes)

    def public_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["api_keys"] = f"{len(self.api_keys)} configured" if self.api_keys else "disabled"
        return d
