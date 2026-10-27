# Routing, profiles and model selection

You normally never name a model. For each page that needs OCR, the router
chooses a **primary model** and a **fallback chain**, and records why.

## Profiles

| Profile | Use it when | Behaviour |
|---|---|---|
| `edge` | Raspberry Pi, Jetson, low RAM | ONNX PP-OCR only, 150 dpi, small detector input, reuses existing OCR layers |
| `fast` | CPU laptops, high volume | PP-OCR on CPU or GPU, 200 dpi, reuses existing OCR layers |
| `balanced` | a GPU with a VLM installed | PP-OCR for simple pages; a document VLM for tables, formulas, hard or low-confidence pages |
| `accurate` | quality first | best installed model first, lighter models as fallback, re-OCRs scanner layers |
| `document` | PDF → structured Markdown | generative document models first (headings, tables, formulas) |
| `server` | an inference server is configured | prefers the remote endpoint (vLLM…), high concurrency |
| `auto` (default) | — | picks one of the above from hardware and installed models |

`auto` resolves to `edge` on ARM boards, `balanced` when a VLM is installed
and runnable on the GPU, `server` when an endpoint is configured, otherwise
`fast`. `textlens setup` saves a default; `TEXTLENS_PROFILE` overrides it.

```python
OCR(profile="balanced")
OCR(quality="high")              # shorthand for accurate under auto
```

## How a model is chosen

1. **Hard constraints** remove models that cannot run:
   runtime not installed · GPU-only model on a CPU machine · VRAM or RAM below
   the spec minimum · remote model without an endpoint · weights missing in
   offline mode · large weights not installed (small ones auto-download) ·
   license policy · latency/memory budgets.
2. **Scoring** ranks the rest: the profile's quality/speed weights × the
   model's tiers, a bonus for generative models on complex pages (tables,
   formulas, difficulty ≥ 0.65) or a penalty on simple ones, a bonus for
   table/formula support where detected, and for already-installed weights.
3. **Measurements beat heuristics.** After `textlens benchmark`, measured
   latencies on *this machine* replace the built-in speed tiers.

The built-in quality/speed tiers are TextLens heuristics, not benchmark
results — measure on your documents.

## Explanations

```bash
textlens ocr scan.pdf --explain
```

```text
Page 2 · scanned · source=ocr · model=ppocrv6-small · confidence=0.991
  OCR reasons: scanned
  Selected: ppocrv6-small  (profile: fast, render: 200 dpi, device: auto)
    • profile 'fast': Fast local OCR with PP-OCR; reuses existing OCR layers in scanned PDFs.
    • auto profile resolved to 'fast' for NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, torch CUDA: no)
    • page needs OCR because: scanned
  Not selected:
    – glm-ocr: profile 'fast' excludes generative models
    – paddleocr-vl: remote model; no endpoint configured (set TEXTLENS_OPENAI_BASE_URL)
```

In Python: `OCR().explain("scan.pdf")` returns the decisions without running
OCR, and every page's `provenance.routing` holds the one that was used.

## Confidence-aware fallback

A page is accepted when it has text, is not garbled, and its confidence is at
or above the profile threshold (`balanced` 0.85, `accurate` 0.90…). Otherwise
the next model runs, and the best result wins. All attempts are stored in
`page.provenance.attempts`.

```python
OCR(min_confidence=0.9, fallback=True)
OCR(fallback_models=["glm-ocr", "ppocrv6-small"])
OCR(fallback=False)
```

Adapter errors (missing runtime, out of memory, server down) also move to the
next model instead of failing the page.

## Budgets and policy

```python
OCR(latency_budget_ms=500)          # skip models estimated/measured slower per page
OCR(memory_budget_gb=4)             # skip models needing more VRAM/RAM
OCR(allowed_licenses=["apache-2.0", "mit"])
OCR(commercial_only=True)           # skip restricted licenses (e.g. AGPL, custom)
```

## Pinning

```python
OCR(model="glm-ocr")                                   # a registered model
OCR(model="glm-ocr", device="cuda:1")
OCR(model="PaddlePaddle/PaddleOCR-VL", backend="openai",
    endpoint="http://gpu-box:8000/v1")                 # anything a server hosts
OCR(model="ppocrv6-small", threads=4, det_limit=1600)  # adapter options
```

A pinned model still gets selective OCR, validation and the result schema.
