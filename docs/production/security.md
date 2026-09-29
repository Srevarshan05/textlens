# Security and privacy

## Privacy model

Your documents do not leave the machine unless you configure a remote
inference endpoint. The router never selects a remote model without one,
and remote backends are explicit (`backend="openai"` / `endpoint=`).

- No telemetry.
- Network access happens only to download model weights you requested (or
  the small default model on first use) — disable with `TEXTLENS_OFFLINE=1`.
- The server keeps uploads in memory and never writes them to disk.
- `TEXTLENS_NO_PERSIST=1` also prevents result caching on disk and
  measurement files.
- Finished jobs are evicted after `TEXTLENS_JOB_TTL` seconds.

## Server hardening checklist

| Control | How |
|---|---|
| Authentication | `TEXTLENS_API_KEYS=…` (`X-API-Key` or `Bearer`), compared in constant time; or `create_app(auth=hook)` for JWT/OIDC/mTLS headers from your gateway |
| Bind address | default `127.0.0.1`; `0.0.0.0` only behind a gateway or with keys (the server warns otherwise) |
| TLS | terminate at a reverse proxy / gateway (nginx, Envoy, cloud LB) |
| Rate limiting | `TEXTLENS_RATE_LIMIT` per client (in-process); enforce globally at the gateway |
| Upload size | `TEXTLENS_MAX_UPLOAD_MB` (413); `TEXTLENS_MAX_PAGES` |
| File types | decided from magic bytes, not names; unknown types → 415; decompression-bomb cap on image pixels |
| URL inputs | off by default; when enabled, only http(s), bounded size, and hosts resolving to private/loopback/link-local addresses are refused (SSRF guard) |
| Local paths | never accepted by default; legacy `image_url` paths only inside `TEXTLENS_ALLOWED_ROOT` |
| CORS | off unless `TEXTLENS_CORS_ORIGINS` is set |
| Backpressure | bounded queue → 503, so floods cannot exhaust memory or trigger runaway GPU autoscaling |
| Headers | `nosniff`, `X-Frame-Options: DENY`, `no-referrer`, `no-store`; `X-Request-ID` on every response |
| Containers | non-root user, read-only root FS in compose/k8s examples, no shell needed at runtime |

### Fixed in 2.0

TextLens 0.x's `/api/v1/ocr` accepted any local file path in `image_url`,
which allowed reading arbitrary files from the server, and imported
`pip install` logic at server start-up. Both are removed.

## Supply chain

- Default model artifacts are pinned by URL **and SHA-256**; mismatches are
  never installed. `textlens models verify <id>` re-checks installed files.
- Hugging Face models can be pinned to a revision in their spec.
- TextLens never runs `pip install` implicitly; `textlens setup
  --install-extras` asks first.
- `trust_remote_code` is used only by adapters whose official usage requires
  it (HunyuanOCR). Review before enabling such models in regulated settings.

## MCP tools

`textlens mcp --workspace DIR` resolves every path against `DIR` and refuses
anything outside it. URL inputs are disabled for MCP tools.

## Licenses

Model licenses vary (see `textlens models list`). `OCR(commercial_only=True)`
or `allowed_licenses=[…]` keep the router away from restricted models.
