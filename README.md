<p align="center">
  <img src="Text-Lens.png" alt="TextLens Framework Banner" width="350"/>
</p>

# TextLens 🔍

[![PyPI Version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/textlens/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TextLens** is a high-performance Python framework for developer-friendly OCR text extraction. Designed for maximum developer flexibility, TextLens features **automatic CUDA driver detection & PyTorch CUDA setup diagnostics**, **multi-model support**, **high-performance batch document processing**, and **1-line REST API endpoint serving** with automatic Swagger UI documentation.

---

## Key Features

- ⚡ **Automated System CUDA Doctor**: Queries system `nvidia-smi`, extracts host GPU CUDA driver version, and provides the exact PyTorch index URL command for any user's PC.
- 🎯 **Multi-Model OCR Engine**: Switch between `glm-ocr`, `lighton-ocr`, `hunyuan-ocr`, and `smolvlm` with a single parameter.
- 🚀 **1-Line REST API Server**: Spin up a production-ready microservice with `textlens.serve(port=8000)` featuring full OpenAPI interactive Swagger documentation (`/docs`).
- 📄 **Multi-Page PDF Processing**: Built-in high-resolution PDF rendering and page-by-page OCR extraction using `pypdfium2`.
- 🗂️ **BatchOCR — High-Performance Batch Processing**: Process entire folders of PDFs and images in parallel with configurable worker threads, automatic retries, structured exports, and a **live real-time monitoring dashboard** at `http://localhost:8765`.
- ⚙️ **Dynamic Runtime Device Switching**: Instantly toggle model execution between `cuda` and `cpu` on the fly.
- 📦 **PyPI Packaging Ready**: Clean standard library structure ready for `pip install textlens`.

---

## Smart System CUDA GPU Setup Doctor

TextLens automatically inspects system hardware so developers and users don't have to manually figure out which PyTorch CUDA wheel matches their graphics driver:

```python
import textlens

# Check CUDA availability programmatically
if textlens.is_cuda_available():
    print("✅ Running with NVIDIA CUDA GPU Acceleration!")
else:
    # Introspect system GPU & get exact installation command for this machine
    sys_cuda = textlens.detect_system_cuda()
    if sys_cuda.has_nvidia_gpu:
        print(f"Detected GPU with CUDA Driver {sys_cuda.system_cuda_version}")
        print("Run this command to enable GPU acceleration:")
        print(f"  {sys_cuda.recommended_install_command}")

# Print detailed hardware diagnostic banner
textlens.print_hardware_status()
```

CLI Command:
```bash
# Run CUDA GPU Doctor Diagnostic
textlens doctor
```

### CPU vs GPU Performance Notice

> [!WARNING]
> **CPU vs GPU Performance Notice**:
> While TextLens fully supports CPU execution as a fallback, running GLM-OCR on CPU will be **significantly slower** than NVIDIA CUDA GPU acceleration, especially when processing multi-page PDFs or high-resolution documents. For production workloads and large document batches, **a CUDA-enabled GPU is strongly recommended**.

---

## Installation

Install TextLens via `pip`:

```bash
pip install textlens
```

Or install from source in editable mode:

```bash
git clone https://github.com/Srevarshan05/textlens.git
cd textlens
pip install -e .
```

---

## Quickstart (Python SDK)

### 1. Basic Text Extraction

```python
from textlens import TextLens

# Auto-detects CUDA GPU or CPU
ocr = TextLens()

# Extract text from local image or URL
text = ocr.read("invoice.png")
print("Extracted Text:\n", text)
```

### 2. High Customization: Markdown Tables, LaTeX Formulas & JSON

```python
from textlens import TextLens

ocr = TextLens()

# Extract Markdown Table
table_md = ocr.extract_table("financial_report.png")
print(table_md)

# Extract LaTeX Math Formulas
formula_latex = ocr.extract_formula("math_sheet.png")
print(formula_latex)

# Extract Structured JSON with Schema
json_output = ocr.extract_json(
    "receipt.jpg",
    schema='{"vendor": "str", "amount": "float", "date": "YYYY-MM-DD"}'
)
print(json_output)
```

### 3. Multi-Page PDF Processing

```python
# Process multi-page PDF document
pdf_pages = ocr.read_pdf("contract.pdf", max_pages=5)

for page in pdf_pages:
    print(f"--- Page {page['page']} of {page['total_pages']} ---")
    print(page["text"])
```

### 4. Dynamic Device Switching

```python
# Dynamically switch execution device at runtime
ocr.switch_device("cpu")   # Moves model weights to CPU
ocr.switch_device("cuda")  # Moves model weights back to CUDA GPU
```

---

## Serving a REST API Endpoint

Deploy an OCR microservice using a single function call:

```python
import textlens

# Launch REST API server on host 127.0.0.1, port 8000
textlens.serve(port=8000)
```

Once launched, access interactive OpenAPI / Swagger UI documentation at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## REST Endpoint API Usage Guide

### 1. Main OCR Endpoint (`POST /api/v1/ocr`)

#### A. File Upload Request (cURL)
```bash
curl -X POST "http://localhost:8000/api/v1/ocr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.png" \
  -F "prompt=Text Recognition:"
```

#### B. File Upload Request (Python `requests`)
```python
import requests

url = "http://localhost:8000/api/v1/ocr"
with open("invoice.png", "rb") as f:
    files = {"file": f}
    data = {"prompt": "Text Recognition:", "max_new_tokens": "512"}
    response = requests.post(url, files=files, data=data)

print(response.json())
```

#### C. JavaScript `fetch` Request
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("prompt", "Text Recognition:");

fetch("http://localhost:8000/api/v1/ocr", {
  method: "POST",
  body: formData
})
  .then(res => res.json())
  .then(data => console.log("OCR Result:", data.text));
```

### 2. JSON Payload Endpoint (`POST /api/v1/ocr/json-payload`)

#### Python `requests`
```python
import requests

url = "http://localhost:8000/api/v1/ocr/json-payload"
payload = {
    "image_url": "https://example.com/sample_invoice.png",
    "prompt": "Text Recognition:",
    "max_new_tokens": 512
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 🗂️ BatchOCR — High-Performance Batch Processing

`BatchOCR` lets you process an **entire folder of documents** (PDFs and images) in parallel with a single API call. TextLens handles queuing, parallel workers, automatic retries, structured output exports, and a **live local monitoring dashboard** — all out of the box.

### Quickstart

```python
from textlens.batch import BatchOCR

batch = BatchOCR(
    model="glm-ocr",      # Any supported TextLens model
    workers=4,            # Parallel worker threads
    output_format="json", # json | markdown | csv | txt
    enable_dashboard=True # Opens http://localhost:8765
)

results = batch.run("./documents/")

for task in results:
    print(f"{task.source_path.name}: {task.status}")
```

> [!NOTE]
> The live monitoring dashboard opens automatically in your browser at **http://localhost:8765** when `enable_dashboard=True` (default). It streams real-time stats without any page refresh.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"glm-ocr"` | OCR model ID to use |
| `workers` | `int` | `4` | Number of parallel worker threads |
| `output_format` | `str` | `"json"` | Export format: `json`, `markdown`, `csv`, `txt` |
| `output_dir` | `str` | `"./batch_output"` | Directory to save OCR results |
| `retries` | `int` | `2` | Retry limit for failed files |
| `dpi` | `int` | `200` | DPI resolution for rendering PDF pages |
| `device` | `str` | `None` (auto) | Force `"cuda"` or `"cpu"` |
| `enable_dashboard` | `bool` | `True` | Launch live monitoring dashboard |
| `dashboard_port` | `int` | `8765` | Dashboard HTTP port |
| `recursive` | `bool` | `True` | Scan subdirectories for files |
| `max_new_tokens` | `int` | `2048` | Max tokens generated per page |

### Processing a Folder

```python
from textlens.batch import BatchOCR

batch = BatchOCR(model="glm-ocr", workers=4)
results = batch.run("./invoices/")
```

**Supported file types**: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`

### Explicit File List

```python
results = batch.run([
    "invoice_jan.pdf",
    "scan_001.png",
    "report_q1.pdf",
])
```

### Callbacks

```python
def on_done(task):
    print(f"✓ {task.source_path.name} — {task.duration_sec:.2f}s")

def on_fail(task):
    print(f"✗ {task.source_path.name} — {task.error}")

results = batch.run("./docs/", on_file_complete=on_done, on_file_failed=on_fail)
```

### Output Formats

Each file produces a separate result in the configured format:

| Format | Extension | Contents |
|---|---|---|
| `json` | `.json` | Full metadata + extracted text |
| `markdown` | `.md` | Formatted Markdown with source info |
| `csv` | `.csv` | Tabular output with one row per file |
| `txt` | `.txt` | Plain extracted text only |

A consolidated **`batch_manifest.json`** is always saved at the end summarising the entire batch job.

```json
{
  "batch_summary": {
    "total_files": 24,
    "completed_files": 23,
    "failed_files": 1,
    "elapsed_time_sec": 142.5,
    "average_speed_fps": 0.16,
    "model_id": "glm-ocr"
  },
  "tasks": [...]
}
```

### Pause, Resume & Cancel

You can control the batch job interactively from Python or via the dashboard:

```python
batch = BatchOCR(model="glm-ocr", workers=4)

# In a separate thread / signal handler:
batch.pause()          # Pauses — in-flight tasks finish, new ones wait
batch.resume()         # Resumes from paused state
batch.cancel()         # Signals workers to stop after current task
batch.retry_failed()   # Re-queues all permanently failed files
```

### Runtime Reconfiguration

Change worker count, output format, or retry limit **while the job is running** — no restart needed:

```python
# Scale up workers dynamically
batch.reconfigure(workers=8)

# Change export format mid-job
batch.reconfigure(output_format="markdown")

# Update retry limit for future tasks
batch.reconfigure(retries=3)
```

### Live Monitoring Dashboard

When `enable_dashboard=True`, a local web server starts automatically and serves a real-time SPA dashboard at **`http://localhost:8765`**.

**Dashboard Features:**

| Feature | Details |
|---|---|
| 📊 **KPI Cards** | Total, Processed, Failed, Queued, Active Workers |
| ⚡ **Speed & ETA** | Files/second throughput and estimated time remaining |
| 🖥️ **System Resources** | CPU %, RAM used/total GB, GPU VRAM used/total GB |
| 📋 **Task List** | Per-file status with duration and error preview |
| 📜 **Live Log Console** | Rolling real-time logs via Server-Sent Events (SSE) |
| ⏸ **Interactive Controls** | Pause, Resume, Cancel, Retry Failed buttons |
| ⚙️ **Live Reconfigure** | Adjust workers and output format without restarting |

The dashboard requires **no external dependencies** — it uses Python's built-in `http.server`.

### CLI Usage

```bash
# Process a folder with 4 workers and open the dashboard
textlens batch ./documents/

# Use a specific model and output format
textlens batch ./invoices/ --model lighton-ocr --format markdown

# Scale to 8 workers, save results to custom directory
textlens batch ./scans/ --workers 8 --output ./results/ --format json

# Process with higher PDF quality (300 DPI)
textlens batch ./pdfs/ --dpi 300 --retries 3

# Disable the dashboard for headless/server environments
textlens batch ./documents/ --no-dashboard

# Use a custom dashboard port
textlens batch ./documents/ --port 9000

# Do not scan subdirectories
textlens batch ./docs/ --no-recursive

# Full example
textlens batch ./documents/ \
  --model glm-ocr \
  --workers 4 \
  --format json \
  --output ./batch_output \
  --retries 2 \
  --dpi 200 \
  --port 8765
```

### Advanced: Custom Queue Backend

`BatchOCR` ships with an in-memory queue by default. The architecture uses a pluggable `BaseBatchQueue` interface, making it straightforward to integrate Redis or any other backend later without changing your application code:

```python
from textlens.batch import BatchOCR, BaseBatchQueue

class MyRedisQueue(BaseBatchQueue):
    # Implement enqueue, dequeue, requeue_failed, etc.
    ...

batch = BatchOCR(
    model="glm-ocr",
    queue_backend=MyRedisQueue(),
)
batch.run("./documents/")
```

### Accessing Results Programmatically

```python
from textlens.batch import BatchOCR, TaskStatus

batch = BatchOCR(model="glm-ocr", workers=2, enable_dashboard=False)
results = batch.run("./docs/")

# Filter results
completed = [t for t in results if t.status == TaskStatus.COMPLETED]
failed    = [t for t in results if t.status == TaskStatus.FAILED]

print(f"Processed: {len(completed)} / {len(results)}")

# Access extracted text
for task in completed:
    print(f"\n{'='*40}")
    print(f"File: {task.source_path.name}")
    print(f"Duration: {task.duration_sec:.2f}s  |  Pages: {task.page_count}")
    print(f"Output: {task.output_path}")
    print(task.result_text[:200])
```

---

## Command Line Interface (CLI)

TextLens includes a built-in CLI command:

```bash
# View all supported models
textlens models

# Install a model
textlens model install glm-ocr

# Run CUDA GPU Doctor Diagnostic & Introspection
textlens doctor

# Run OCR on an image file or PDF
textlens read document.png --prompt "Text Recognition:"
textlens read report.pdf

# Batch process an entire folder
textlens batch ./documents/ --model glm-ocr --workers 4

# Launch REST API endpoint server
textlens serve --port 8000
```

---

## Testing & PyPI Publishing Guide

### Running Automated Tests
Run unit tests using `pytest`:

```bash
python -m pytest tests/ -q
```

### Building & Pushing to PyPI

1. Install build tools:
   ```bash
   pip install build twine
   ```

2. Build source distribution and wheel:
   ```bash
   python -m build
   ```

3. Upload package to PyPI:
   ```bash
   twine upload dist/*
   ```

---

## License

MIT License © TextLens Contributors
