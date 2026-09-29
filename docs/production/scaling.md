# Scaling, batching and caching

The same `OCR` API runs from a script to a cluster. What changes is where
work queues, where models live, and how pages reach the GPU.

```text
1  script           OCR()("scan.pdf")
2  batch            OCR().batch(folder, workers=N)  ·  textlens batch
3  service          textlens serve  (job pool, backpressure, metrics)
4  inference tier   TextLens API/workers ──► vLLM / SGLang serving a document VLM
5  decoupled        gateway ─► API ─► queue ─► TextLens workers ─► inference server ─► result store
```

## Know where time goes first

```bash
textlens profile representative.pdf --repeat 3
```

On most document corpora the biggest win is not a faster model but **not
running one**: selective OCR skips native pages, and the router keeps simple
scanned pages on the fast engine.

## Level 2: batch

```python
results = OCR(profile="fast").batch("./inbox", workers=4)
```

- Workers share one loaded model (the backend pool); GPU models serialise
  inference per model, ONNX sessions run concurrently.
- Failures come back as exception objects; the batch continues.
- `textlens batch ./inbox -o out/ --resume` skips files that already have
  output — rerun after a crash or cancellation.
- Identical files are recognised by content hash (result cache).

## Level 3: service

`textlens serve` puts a bounded job pool in front of inference:

- `TEXTLENS_WORKERS` concurrent jobs (1 per GPU for local VLMs; more for ONNX
  on multi-core CPUs or remote backends).
- `TEXTLENS_MAX_QUEUE` waiting jobs; beyond that requests get **503 +
  Retry-After** (backpressure) rather than piling up in memory.
- async jobs (`async=true`) with polling, cancellation, retries (non-input
  errors), TTL eviction; NDJSON page streaming for long documents.

Scale horizontally with more replicas behind a load balancer. Each replica
loads its own models; size replicas to your GPUs.

## Level 4: an inference server tier

For heavyweight VLMs, run them in an inference server with continuous
batching and let TextLens route to it:

```bash
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code
TEXTLENS_OPENAI_BASE_URL=http://vllm:8000/v1 TEXTLENS_PROFILE=server TEXTLENS_WORKERS=8 textlens serve
```

TextLens keeps selective OCR, validation, provenance and the result schema;
the inference server batches pages across all TextLens replicas. The OpenAI
backend sends a document's pages concurrently (`concurrency` option) so the
server sees batches, retries 429/5xx with backoff, and uses token logprobs
as a confidence proxy where the server supports them.

## Level 5: decoupled queue

When ingestion and inference must scale independently (bursty uploads,
scale-to-zero GPU pools), put a queue between them:

```text
client ─► gateway (auth, rate limits) ─► API replicas ─► queue (Redis/SQS/Kafka)
      ─► TextLens workers (CPU: inspect/native/render; GPU: OCR) ─► inference server ─► result store
```

TextLens's extension points for this:

- `textlens.runtime.jobs.JobStore` — implement for Redis/SQL so API replicas
  and workers share job state (the in-memory store is the default).
- `textlens.runtime.batching.DynamicBatcher` — the collector pattern: block
  for the first item, collect for `window_ms` up to `max_batch`, run one
  `recognize()` call, return a future per item. One batch per GPU at a time
  keeps VRAM predictable.
- `suggest_batch_size(spec)` — memory-aware batch size from free VRAM/RAM.
- Workers are ordinary Python processes calling `OCR(...)`; scale them on
  queue depth (e.g. KEDA) and GPU pools separately from the API.

This mirrors the architecture taught in the
[Neural Maze production OCR course](https://github.com/neural-maze/production-ocr-course)
(queue decoupling, collector batching, autoscaling on queue depth). TextLens
does not ship Redis or Kubernetes clients in its core; they belong in your
deployment layer.

## Caching

| Layer | Default | Notes |
|---|---|---|
| Model weights | `~/.cache/textlens/models` | shared volume in containers |
| Loaded models | process-wide pool | one copy per (model, device, options) |
| Results | in-memory LRU | key = document SHA-256 + config hash (model, profile, pages, DPI, version…) |

```python
OCR(cache="disk")                 # persist results under TEXTLENS_CACHE_DIR
OCR(cache="off")
ocr(file, cache=False)            # bypass for one call
```

`TEXTLENS_NO_PERSIST=1` forbids disk caching. `textlens cache stats|clear`.

## Large PDFs

- Pages stream; memory is flat in page count.
- PDFium is single-threaded (a process-wide lock) — parallelise documents
  with processes/replicas, not threads.
- `TEXTLENS_MAX_PAGES` and `TEXTLENS_MAX_FILE_MB` bound work per request.
