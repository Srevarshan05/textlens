# REST server

```bash
pip install "textlens-ocr[server]"
textlens serve                                   # http://127.0.0.1:8000, docs at /docs
TEXTLENS_API_KEYS=k1,k2 textlens serve --host 0.0.0.0 --workers 2 --warmup
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ocr` | image/PDF/DOCX/PPTX → result (`format=json|text|markdown|html`) |
| POST | `/document` | structure-first processing (defaults to Markdown) |
| POST | `/extract` | `schema` → JSON fields |
| POST | `/inspect` | PDF classification / image signals, no OCR |
| POST | `/batch` | many files → async job |
| GET | `/jobs/{id}` | job status and progress |
| GET | `/jobs/{id}/result` | job result (202 while running) |
| DELETE | `/jobs/{id}` | cancel |
| GET | `/models` | catalog with install/runtime status |
| GET | `/health` | liveness |
| GET | `/ready` | readiness: model routable, warm, queue not full (503 otherwise) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/system` | hardware, runtimes, loaded models, settings |
| POST | `/v1/chat/completions`, GET `/v1/models` | OpenAI-compatible facade |
| * | `/api/v1/*` | 0.x endpoints (compatibility) |

### Examples

```bash
curl -H "X-API-Key: k1" -F file=@scan.pdf localhost:8000/ocr
curl -H "X-API-Key: k1" -F file=@scan.pdf -F format=markdown -F pages=1-3 localhost:8000/ocr
curl -H "X-API-Key: k1" -F file=@invoice.pdf -F 'schema={"total":"float","date":"date"}' localhost:8000/extract

# async
curl -H "X-API-Key: k1" -F file=@big.pdf -F async=true localhost:8000/ocr     # → {"job_id": "…"}
curl -H "X-API-Key: k1" localhost:8000/jobs/<id>
curl -H "X-API-Key: k1" localhost:8000/jobs/<id>/result

# stream pages as NDJSON as soon as each finishes
curl -N -H "X-API-Key: k1" -F file=@big.pdf -F stream=true localhost:8000/ocr
```

Per-request routing: form fields `profile`, `model`, `ocr` (`auto|force|off`),
`pages`.

### OpenAI-compatible clients

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="k1")
r = client.chat.completions.create(model="textlens", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}},
    {"type": "text", "text": "markdown"}]}])
print(r.choices[0].message.content)
```

`model` may be `textlens`, `profile:<name>` or a model id.

## Configuration

All settings are environment variables (see
`textlens/serving/settings.py`), with CLI flags for common ones:

| Variable | Default | Meaning |
|---|---|---|
| `TEXTLENS_HOST` / `TEXTLENS_PORT` | 127.0.0.1 / 8000 | bind address |
| `TEXTLENS_API_KEYS` | — | comma-separated keys; auth is off when empty |
| `TEXTLENS_PROFILE`, `TEXTLENS_MODEL`, `TEXTLENS_DEVICE` | auto | routing defaults |
| `TEXTLENS_OPENAI_BASE_URL` | — | inference server for served models |
| `TEXTLENS_WORKERS` | 1 | concurrent inference jobs |
| `TEXTLENS_MAX_QUEUE` | 64 | waiting jobs before 503 |
| `TEXTLENS_MAX_UPLOAD_MB` | 50 | per-file limit |
| `TEXTLENS_MAX_PAGES` | 500 | pages per document |
| `TEXTLENS_RATE_LIMIT` / `TEXTLENS_RATE_BURST` | 0 / 10 | per-client token bucket (req/s) |
| `TEXTLENS_SERVER_ALLOW_URLS` | 0 | accept `url=` inputs (SSRF-guarded) |
| `TEXTLENS_NO_PERSIST` | 0 | never persist results (cache memory-only) |
| `TEXTLENS_CORS_ORIGINS` | — | allowed browser origins |
| `TEXTLENS_JOB_TTL` | 3600 | seconds finished jobs are kept |
| `TEXTLENS_SYNC_TIMEOUT` | 300 | max wait for synchronous requests |
| `TEXTLENS_WARMUP` | 0 | load the default model at startup |
| `TEXTLENS_ALLOWED_ROOT` | — | directory legacy `image_url` paths may read |
| `TEXTLENS_LOG_FORMAT` | text | `json` for structured logs |

## Behaviour under load

- Uploads stay in memory (bounded); nothing is written to disk.
- Inference runs in a bounded job pool. Synchronous requests wait for their
  job; when the queue is full the server answers **503 + Retry-After** instead
  of exhausting memory.
- One loaded copy of each model is shared by all workers.

For GPU VLMs keep `TEXTLENS_WORKERS=1` per GPU; ONNX PP-OCR and remote
backends tolerate more. See [Scaling](../production/scaling.md) and
[Security](../production/security.md).

## Embedding in your own FastAPI app

```python
from textlens.serving.app import create_app
from textlens.serving.settings import ServerSettings

app = create_app(ServerSettings(api_keys=("k1",), workers=2),
                 auth=lambda request: request.headers.get("x-user"))   # custom auth hook
```
