# Changelog

## 2.0.0a1 — 2026-09-29

TextLens 2.0 turns the multi-model OCR wrapper into an OCR infrastructure
layer. See [docs/migration.md](docs/migration.md) for upgrade notes and
[docs/design/audit-2.0.md](docs/design/audit-2.0.md) for the rationale.

### Added
- `OCR()(source) -> Result`: unified result schema (pages, blocks, lines, words,
  tables, formulas, boxes, confidence, provenance) with text/Markdown/HTML/JSON/CSV
  exports, RAG chunks with citations, and artifact bundles.
- PDF intelligence: per-page inspection with reason codes, native positioned
  extraction with reading order and ruled tables, garbled-text detection,
  selective OCR (`ocr="auto"|"force"|"off"`), OCR-layer reuse, streaming, lazy
  `textlens.load()` documents, DOCX/PPTX input.
- Default engine PP-OCRv6 Small on ONNX Runtime (numpy + Pillow, no OpenCV/PyTorch),
  SHA-256-pinned artifacts.
- Router with profiles (`auto`, `edge`, `fast`, `balanced`, `accurate`,
  `document`, `server`), hardware/runtime constraints, latency/memory budgets,
  license policy, explanations, confidence-validated fallback chains.
- Spec-driven model registry (capabilities, hardware, VRAM, license, limitations),
  plugin entry points, served models via OpenAI-compatible endpoints (vLLM…).
- CLI: `ocr`, `document`, `inspect` (+ overlays), `extract`, `models
  list|search|install|info|remove|verify|path`, `doctor` (+ `--deep`, `--json`),
  `setup`, `serve`, `mcp`, `benchmark`, `profile`, `anpr`, `cache`.
- Server: `/ocr`, `/document`, `/extract`, `/inspect`, `/batch`, `/jobs`, `/models`,
  `/health`, `/ready`, `/metrics`, `/system`, OpenAI-compatible `/v1/chat/completions`;
  API keys/auth hook, rate limiting, upload limits, SSRF guard, async jobs,
  NDJSON streaming, backpressure.
- Runtime: result cache, job manager, dynamic batcher, memory-aware batch sizing,
  structured JSON logging, Prometheus metrics.
- Evaluation: CER/WER/exact/NED/table-cell metrics, benchmark runner on your data
  (measurements feed routing), profiler, visual overlays.
- ANPR pipeline: detection → rectification → recognition → region-format
  validation/correction → tracking with temporal voting.
- MCP server, Docker (CPU/GPU/Jetson), Compose, Kubernetes reference, CI.

### Changed
- Python ≥ 3.10. Core deps: pillow, pypdfium2, numpy, onnxruntime.
- Default model `ppocrv6-small` (was `glm-ocr`); GLM decoding is greedy.
- BatchOCR shares one engine across workers; routed model by default; `--resume`.
- Spec-driven doctor; corrected VRAM/CPU metadata.

### Fixed
- Arbitrary local file read via the server's `image_url`.
- Implicit `pip install` at import/initialisation.
- One model copy per BatchOCR worker.
- Broken Hugging Face discovery (`disable_progress_bars` misuse).
- Duplicate `ModelRegistry.is_registered`; `textlens read` ignoring `--model`.
- Windows console crashes on non-cp1252 output; directories created at import.

## 0.1.1 and earlier

See git history.
