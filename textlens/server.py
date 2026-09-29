"""
textlens.server
───────────────
Backwards-compatible entry point for the REST server.

The server lives in :mod:`textlens.serving.app` since TextLens 2.0.  This
module keeps the 0.x API (``create_app(engine=…)``, ``serve(host, port)``)
working.  Unlike 0.x it no longer installs packages at import time — install
the server extra instead: ``pip install "textlens-ocr[server]"``.
"""

from __future__ import annotations

from typing import Any, Optional

from textlens.serving.app import create_app as _create_app
from textlens.serving.app import serve as _serve
from textlens.serving.settings import ServerSettings


def create_app(engine: Any = None, settings: Optional[ServerSettings] = None, **kwargs: Any) -> Any:
    """Build the FastAPI app; ``engine`` may be a 0.x ``TextLens`` instance."""
    return _create_app(settings=settings, engine=engine, **kwargs)


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, engine: Any = None) -> None:
    """Run the server (0.x signature)."""
    _serve(host=host, port=port, engine=engine)


__all__ = ["create_app", "serve", "ServerSettings"]
