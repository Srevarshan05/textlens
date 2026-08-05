<p align="center">
  <img src="Text-Lens.png" alt="TextLens Framework Banner" width="350"/>
</p>

# TextLens 🔍

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/textlens/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TextLens** is a high-performance Python framework for developer-friendly OCR text extraction powered by GLM-OCR (`zai-org/GLM-OCR`). Designed for maximum developer flexibility, TextLens features **automatic CUDA driver detection & PyTorch CUDA setup diagnostics**, **high customization for tables, formulas, and structured JSON**, and **1-line REST API endpoint serving** with automatic Swagger UI documentation.

---

## Key Features

- ⚡ **Automated System CUDA Doctor**: Queries system `nvidia-smi`, extracts host GPU CUDA driver version (e.g. `13.1`, `12.4`, `11.8`), and provides the exact PyTorch index URL command for any user's PC.
- 🎯 **GLM-OCR Vision Engine Core**: High accuracy extraction for standard text, complex Markdown tables, LaTeX mathematical equations, and key-value JSON objects.
- 🚀 **1-Line REST API Server**: Spin up a production-ready microservice with `textlens.serve(port=8000)` featuring full OpenAPI interactive Swagger documentation (`/docs`).
- 📄 **Multi-Page PDF Processing**: Built-in high-resolution PDF rendering and page-by-page OCR extraction using `pypdfium2`.
- ⚙️ **Dynamic Runtime Device Switching**: Instantly toggle model execution between `cuda` and `cpu` on the fly (`ocr.switch_device("cpu")`).
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

## Command Line Interface (CLI)

TextLens includes a built-in CLI command:

```bash
# Run CUDA GPU Doctor Diagnostic & Introspection
textlens doctor

# Run OCR on an image file or PDF
textlens read document.png --prompt "Text Recognition:"

# Launch REST API endpoint server
textlens serve --port 8000
```

---

## Testing & PyPI Publishing Guide

### Running Automated Tests
Run unit tests using python `unittest`:

```bash
python -m unittest discover -s tests -p "test_*.py"
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
