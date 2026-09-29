"""
textlens.serving.security
─────────────────────────
Authentication hooks, rate limiting and request-size enforcement.

* **API keys** — ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``,
  compared in constant time.  Keys come from ``TEXTLENS_API_KEYS``.
* **Custom auth** — pass ``auth=callable(request) -> principal | None`` to
  :func:`textlens.serving.app.create_app` to plug in JWT/OIDC/mTLS checks;
  return ``None`` to reject.  A gateway (APIM, Kong, Envoy) can do this in
  front instead — see ``docs/production/security.md``.
* **Rate limiting** — per-principal token bucket (in-process).  Behind
  several replicas, enforce limits at the gateway.
"""

from __future__ import annotations

import hmac
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

AuthHook = Callable[[Any], Optional[str]]


def api_key_principal(request: Any, keys: Tuple[str, ...]) -> Optional[str]:
    """Return a principal for a valid key, ``None`` otherwise."""
    supplied = request.headers.get("x-api-key")
    if not supplied:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if not supplied:
        return None
    for i, key in enumerate(keys):
        if hmac.compare_digest(supplied.encode(), key.encode()):
            return f"key-{i}"
    return None


class RateLimiter:
    """Token bucket per client: ``rate`` tokens/second, ``burst`` capacity."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = float(rate)
        self.burst = max(1, int(burst))
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> Tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        if self.rate <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(client, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, now)
                return True, 0.0
            self._buckets[client] = (tokens, now)
            if len(self._buckets) > 10_000:  # bound memory under many clients
                for k in list(self._buckets)[:5_000]:
                    self._buckets.pop(k, None)
            return False, (1.0 - tokens) / self.rate


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
