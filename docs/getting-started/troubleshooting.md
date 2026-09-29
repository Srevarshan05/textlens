# Troubleshooting

Start with:

```bash
textlens doctor          # hardware, runtimes, model readiness, fix commands
textlens doctor --deep   # also imports PyTorch to confirm CUDA works
```

Errors carry a `code` and a `Hint:` line with the next step. With the CLI,
`--verbose --verbose` prints tracebacks.

## Installation

**`pip install textlens` fails ("No matching distribution")**
The package name is `textlens-ocr`: `pip install textlens-ocr`.

**`onnxruntime` fails to install on a Raspberry Pi**
ONNX Runtime publishes no 32-bit ARM wheels. Use a 64-bit OS (Raspberry Pi
OS 64-bit, Ubuntu arm64). See [Raspberry Pi](../deployment/raspberry-pi.md).

**`BackendUnavailableError: GLM-OCR needs torch, transformers`**
Install the GPU extra: `pip install "textlens-ocr[gpu]"`, after a CUDA
torch build if you have an NVIDIA GPU.

**GLM-OCR / LightOnOCR / HunyuanOCR fail to import model classes**
They need `transformers>=5`. `textlens doctor` warns when your version is
older. `pip install -U transformers`.

## GPU

**NVIDIA GPU present, but TextLens runs on CPU**
Your torch is a CPU build (the default on Windows PyPI). `textlens doctor`
prints the matching `pip install torch --index-url …/cuXXX` command.

**PP-OCR should use the GPU**
The default `onnxruntime` wheel is CPU-only. Replace it:
`pip uninstall -y onnxruntime && pip install onnxruntime-gpu`. Never install
both (doctor warns about this).

**CUDA out of memory**
Use a smaller model or profile, or cap memory: `OCR(memory_budget_gb=4)`.
Models that are too large for the GPU but practical on CPU are placed on the
CPU automatically.

## Results

**A scanned PDF returns empty text**
Run `textlens inspect file.pdf`. If pages show `needs_ocr=no` with almost no
text, force OCR: `OCR(ocr="force")` or `--ocr force`.

**Garbled text from a PDF (random letters, `$` between words, boxes)**
The text layer is broken. TextLens detects most cases
(`suspected_garbled_text`) and OCRs those pages. If one slips through,
use `--ocr force` for that document and please report the file.

**Reading order is wrong in columns**
Check the overlay: `textlens inspect doc.pdf --overlay out/` numbers blocks in
reading order. Tabular layouts are read row-by-row on purpose.

**Low confidence pages**
`result.pages_needing_review()` lists them. Enable fallback to a stronger model:
`OCR(profile="balanced")` with a VLM installed, or `fallback_models=[...]`.

## Performance

**First call is slow**
It includes model download (once) and loading (~0.2–1 s for PP-OCR). Servers
can load at startup with `textlens serve --warmup`.

**Repeated runs get slower on a laptop**
On hybrid Intel CPUs (P- and E-cores), Windows may move a sustained background
process to efficiency cores. Pin inference threads to the performance-core
count, e.g. `OCR(threads=6)`, use a high-performance power plan, and prefer
running in the foreground. Measure with `textlens profile file.pdf --repeat 3`.

**Edge device runs out of memory**
`OCR(profile="edge", low_memory=True)` disables ONNX Runtime's memory arena
(smaller footprint, ~3× slower detection).

## Server

**401 Unauthorized** — send `X-API-Key: <key>` or `Authorization: Bearer <key>`.
**413** — upload above `TEXTLENS_MAX_UPLOAD_MB`.
**415** — not a supported file type (decided from content, not extension).
**429** — rate limit; honour `Retry-After`.
**503** — queue full (backpressure); retry after `Retry-After` seconds, or raise
`TEXTLENS_WORKERS` / `TEXTLENS_MAX_QUEUE`.
**URL inputs rejected** — URL fetching is off by default on servers
(`--allow-urls`), and private/loopback hosts are always refused (SSRF guard).
