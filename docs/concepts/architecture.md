# Architecture

TextLens is not a model. It is the layer around models: it decides **what
needs OCR**, **which engine should do it on this hardware**, **whether the
answer is good enough**, and returns **one schema** no matter which engine
ran.

```text
input (path · URL · bytes · stream · PIL · numpy · DOCX · PPTX)
  │  inputs/source.py   magic-byte sniffing, size limits, content hash
  ▼
document analysis
  │  PDF: documents/pdf/inspector.py   per-page kind + OCR reason codes (ms, no rendering)
  │  image: core/difficulty.py         contrast, blur, density, ruling lines
  ▼
per page ──► native text usable? ──yes──► documents/pdf/native.py
  │                                       words → lines → paragraphs → reading order, tables
  │                                       core/quality.py rejects garbled layers → OCR instead
  │  no
  ▼
render (only the pages that need it)
  ▼
core/router.py   profile + hardware + installed models + budgets + page signals
  │              → primary model, fallback chain, explanation
  ▼
backends/*       adapter.recognize(images) → PageOCR (pixel space)
  │              onnx/ppocr (edge/default) · transformers VLMs · openai_compat (vLLM…)
  ▼
validation       confidence ≥ threshold, not empty, not garbled — else next model
  ▼
core/result.py   Page / Block / Line / Word / Table / Formula + provenance, in page units
  ▼
exports          text · Markdown · HTML · JSON · CSV · RAG chunks · artifact bundle
```

## Why each stage exists

**Inspect before OCR.** Roughly half of real-world PDFs have a usable text
layer. Rendering and OCRing those pages wastes GPU time and loses exact text.
The inspector reads PDF objects (text, images, paths, forms) and the PDFium
text layer in milliseconds per page and gives every page a reason code.

**Check the text layer's quality.** A text layer can exist and still be
garbage: broken ToUnicode maps, CID fonts, scanner OCR layers under an image.
TextLens detects those and OCRs the page instead of returning mojibake.

**Route per page.** A 200-page report with 3 scanned tables should not run a
3 B-parameter VLM on 200 pages. The router sends simple pages to a fast engine
and complex or low-confidence pages to a stronger one, within your hardware
and budgets.

**Validate, then fall back.** Engines report confidence differently. TextLens
accepts a page only if it is non-empty, not garbled and above the profile's
threshold; otherwise the next model in the chain runs. Every attempt is kept
in provenance.

**One schema.** Application code never branches on the engine. Native text,
PP-OCR lines and VLM Markdown all become the same blocks with boxes (where the
engine provides them), confidence and provenance.

## Packages

| Package | Responsibility |
|---|---|
| `textlens.core` | `OCR` engine, pipeline, router, result schema, exports, quality checks |
| `textlens.inputs` | input normalisation and image handling |
| `textlens.documents` | PDF inspector, native extraction, layout, DOCX/PPTX, `Document` |
| `textlens.backends` | adapter contract, backend pool, ONNX edge engine, VLM and remote adapters |
| `textlens.models` | registry (specs), verified downloads, doctor, hardware, discovery |
| `textlens.runtime` | system inspection, result cache, jobs, dynamic batching |
| `textlens.serving` | FastAPI server, OpenAI-compatible facade, security, MCP |
| `textlens.evaluation` | metrics, benchmark runner, profiler, overlays |
| `textlens.anpr` | number-plate pipeline (independent of the core) |
| `textlens.cli` | the `textlens` command |

## Design rules

- **Lazy everything.** `import textlens` imports no ML runtime. Adapters import
  torch/onnxruntime only when a model loads (a test enforces this).
- **No hidden side effects.** No `pip install`, no directories created at import,
  no network access unless a model is missing and downloads are allowed.
- **One copy of each model per process.** The backend pool shares loaded models
  across `OCR` instances, batch workers and server requests.
- **PDFium is not thread-safe.** All PDFium calls run under one lock; parallelism
  across documents belongs in processes (or more server replicas).
- **Model-specific code lives in adapters.** The core only sees `ModelSpec`
  capabilities and the `OCRBackend` contract.

## Scaling the same API

```text
student script      OCR()("scan.pdf")
application         OCR(profile="balanced")
service             textlens serve  (jobs, backpressure, metrics, auth)
production          gateway → TextLens API replicas → inference server (vLLM) → result store
```

The call site does not change. See [Scaling](../production/scaling.md).
