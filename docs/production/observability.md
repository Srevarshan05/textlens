# Observability

## Metrics (`GET /metrics`, Prometheus format)

| Metric | Type | Labels |
|---|---|---|
| `textlens_http_requests_total` | counter | route, method, status |
| `textlens_http_request_seconds` | histogram | route |
| `textlens_pages_total` | counter | source (`native`, `ocr`, `vlm`, `fused`, `empty`) |
| `textlens_jobs_total` | counter | status, kind |
| `textlens_jobs_active` | gauge | — |
| `textlens_job_seconds` | histogram | kind |
| `textlens_queue_pending` | gauge | — |
| `textlens_models_loaded` | gauge | — |

`textlens_pages_total{source="native"}` vs `{source="ocr"}` shows how much
work selective OCR saves. Alert on `textlens_queue_pending` approaching
`workers + max_queue` (503s imminent). `/metrics` is public by default for
scrapers (`TEXTLENS_METRICS_PUBLIC=0` to require a key).

The registry is dependency-free (`textlens.observability.metrics`) and usable
from your own code:

```python
from textlens.observability import metrics
metrics.counter("myapp_invoices_total", "Invoices processed").inc(vendor="acme")
```

## Structured logs

```bash
TEXTLENS_LOG_FORMAT=json TEXTLENS_LOG_LEVEL=INFO textlens serve
```

One JSON object per line with `ts`, `level`, `logger`, `msg` and fields such
as `request_id`, `path`, `status`, `ms`. TextLens configures only the
`textlens` logger, never the root logger.

## Health

- `GET /health` — liveness (process up).
- `GET /ready` — readiness: warm-up finished, a model is routable, queue not
  full. Returns 503 with `problems` otherwise.
- `GET /system` — hardware, runtimes, loaded models, effective settings.

## Per-result provenance

Every page records model, backend, device, DPI, timings and all fallback
attempts; every result records the config hash and document hash. Log
`result.provenance` (small) alongside business events to explain any output
later.

## Diagnosing a slow deployment

```bash
textlens profile sample.pdf --repeat 3     # stage timings, load time, peak RAM/VRAM
textlens doctor --json                     # runtime/provider problems
```
