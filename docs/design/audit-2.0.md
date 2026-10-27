# TextLens 2.0 — audit, gap analysis and migration plan

Written 2026-09-29 before the 2.0 rewrite, updated with what was implemented.
Kept as the record of *why* the architecture changed.

## 1. The 0.x architecture

```text
OCR(model) ─► ModelRegistry (4 static entries) ─► auto-download (HF snapshot, unpinned)
          └─► _resolve_backend (if-chain) ─► GLM / SmolVLM / LightOn / Hunyuan backend
                 └─► every input → rasterise every page → VLM → str ("--- Page N ---" joined)
TextLens (sdk.py)   GLM-only, eager torch; used by `textlens read` and the REST server
BatchOCR            thread pool; each worker built its own OCR() → exporters + web dashboard
models/             metadata, registry, cache, downloader, manager, doctor, nvidia-smi hardware, HF discovery
```

Baseline: 103 tests passing, 5 failing.

## 2. What was good (kept)

Lazy public imports (with a regression test) · a frozen metadata catalog ·
fast `nvidia-smi` hardware detection without importing torch · the model cache
abstraction · BatchOCR's queue interface, controls and dashboard · a
structured exception base · a unified input loader · VLM backends following
official Hugging Face usage.

## 3. What was broken

| Issue | Impact | Resolution in 2.0 |
|---|---|---|
| `OCR()` defaulted to a 1.3 B VLM while torch/transformers/huggingface_hub were not dependencies | the documented basic path failed on a fresh install | default engine is PP-OCRv6 on ONNX Runtime (core deps); VLMs are routed to when installed |
| `ensure_dependencies(auto_install=True)` ran `pip install torch…` at server import and in `TextLens()` | silent environment mutation; could replace CUDA torch with CPU torch | removed implicit calls; `TextLens(auto_fix_dependencies=False)` by default |
| Server `image_url` accepted any local path; CORS `*` with credentials; no limits/auth | arbitrary file read | server rebuilt: API keys/auth hook, limits, magic-byte validation, SSRF guard, `TEXTLENS_ALLOWED_ROOT` |
| BatchOCR built one model per worker thread | N× VRAM → OOM | shared engine + process-wide backend pool |
| Every PDF page rasterised through a VLM | slow, costly, loses exact native text | inspector + selective OCR + native extraction |
| Results were strings | no boxes, confidence, provenance | unified `Result` schema |
| HF discovery used `disable_progress_bars()` as a context manager | discovery broken, 5 failing tests | fixed |
| `ModelRegistry.is_registered` defined twice; `textlens read` ignored `--model` | silent wrong behaviour | fixed; CLI rebuilt |
| GLM sampling at temperature 0.7 by default | non-reproducible OCR | greedy decoding by default |
| Metadata drift, unpinned unverified downloads | wrong advice, irreproducible installs | spec-driven registry; SHA-256-pinned artifacts |
| Windows console crash on non-cp1252 output; import-time mkdir; hard-coded paths | fragile | UTF-8-safe CLI streams; lazy directories; paths from config |

Also flagged to the maintainer: personal PDFs tracked in git; the PyPI name
`textlens` is blocked by the existing `text-lens` project (hence
`textlens-ocr`).

## 4. Component map (implemented)

| Spec requirement | Module |
|---|---|
| unified OCR API | `core/engine.py` (`OCR`) |
| unified result schema + exports + chunks | `core/result.py`, `core/export.py`, `core/markdown.py` |
| model registry with capabilities/hardware/license | `models/specs.py` (+ `registry.py` compat view) |
| routing, profiles, budgets, explanations | `core/router.py` |
| hardware/runtime detection | `runtime/system.py`, `models/hardware.py` |
| PDF classification + selective OCR | `documents/pdf/inspector.py`, `core/pipeline.py` |
| native extraction, reading order, tables | `documents/pdf/native.py`, `documents/layout.py` |
| garbled-text detection | `core/quality.py` |
| confidence validation + fallback | `core/pipeline.py` (`ocr_image`) |
| edge OCR (ONNX, no OpenCV) | `backends/onnx/` |
| local VLMs | `backends/transformers_vlm.py` (wrapping 0.x backends) |
| served models (vLLM etc.) | `backends/openai_compat.py` |
| Document API | `documents/document.py` |
| DOCX / PPTX | `documents/office.py` |
| result cache | `runtime/cache.py` |
| jobs, backpressure, dynamic batching | `runtime/jobs.py`, `runtime/batching.py` |
| server, OpenAI facade, MCP | `serving/` |
| benchmark, profile, overlays | `evaluation/` |
| ANPR | `anpr/` |
| models / doctor / setup CLI | `cli/` |
| Docker, k8s, CI | `deploy/`, `.github/workflows/ci.yml` |

## 5. Dependency strategy

Core: `pillow`, `pypdfium2`, `numpy`, `onnxruntime` (skipped on 32-bit ARM).
Extras: `gpu`, `documents`, `server`, `vllm`, `anpr`, `edge`, `mcp`, `ui`,
`catalog`, `dev`, curated `all`; 0.x extras kept as aliases. Nothing is
installed implicitly.

## 6. Model strategy

Few engines, chosen for coverage: PP-OCRv6 (edge/default), GLM-OCR,
LightOnOCR-2, HunyuanOCR, SmolVLM (local), PaddleOCR-VL and DeepSeek-OCR/-2
(served), MinerU2.5 and Surya (catalog only, license-sensitive). ANPR uses
open-image-models + fast-plate-ocr (MIT). Licenses verified from model cards.

## 7. Phases

| Phase | Scope | Status |
|---|---|---|
| P0 | bug fixes above | done |
| P1 | config, errors, Result, edge ONNX backend | done |
| P2 | inspector, native extraction, selective OCR, Document | done |
| P3 | router, profiles, fallback, difficulty | done |
| P4 | models/doctor/setup/CLI | done |
| P5 | cache, batching, jobs, BatchOCR on shared engine | done |
| P6 | server, OpenAI facade, Docker | done (GPU/Jetson images untested) |
| P7 | benchmark, profile, overlays | done |
| P8 | ANPR | done |
| P9 | remote/vLLM backend, extraction, MCP | done (MCP not integration-tested) |
| P10 | docs, examples, CI | done |

## 8. Known limitations / next steps

- Borderless native tables are returned as tab-separated lines.
- Local VLM adapters report no confidence (generation scores could provide one).
- VLM outputs have no boxes; a layout detector (e.g. PP-DocLayout) could crop
  regions before generation, as in layout-first production pipelines.
- Redis/SQL `JobStore`, OpenTelemetry tracing and a Helm chart are extension
  points, not shipped.
- GPU and Jetson Docker images and the MCP server need validation on real
  hardware/clients.
