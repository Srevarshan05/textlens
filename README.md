<p align="center">
  <img src="Text-Lens.png" alt="TextLens Logo" width="380" />
</p>

# textlens

[![PyPI Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/Srevarshan05/textlens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg)](https://developer.nvidia.com/cuda-zone)

> **A high-performance, developer-friendly Python OCR framework wrapping GLM-OCR (`zai-org/GLM-OCR`).**  
> Features automatic GPU (PyTorch CUDA) / CPU hardware detection, single-line REST API server hosting, multi-page PDF processing, table-to-Markdown formatting, math LaTeX rendering, structured JSON schema extraction, and dynamic runtime device migration.

---

## 🚀 Key Features

- **⚡ Hardware Auto-Detection**: Instant PyTorch CUDA / CPU introspection (`torch.cuda.is_available()`, GPU model name, total & allocated VRAM, CUDA runtime status).
- **📄 Complete Document Reading**: Read text from local images (`PNG`, `JPG`, `WEBP`), multi-page `PDF` documents, or remote HTTP/HTTPS URLs.
- **📊 Specialized Extractions**: Built-in methods for **Tables → Markdown**, **Formulas → LaTeX**, and **Documents → Structured JSON**.
- **🔄 Dynamic Device Migration**: Switch model execution dynamically between GPU (`cuda`) and CPU (`cpu`) at runtime with cache optimization (`torch.cuda.empty_cache()`).
- **🌐 One-Line REST API Server**: Spin up a FastAPI REST API microservice with interactive OpenAPI Swagger UI (`/docs`).
- **🛠️ Built-in CLI**: Run hardware diagnostics, perform terminal OCR, or host the REST server directly from command line.

---

## 📦 Installation

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

## ⚙️ Hardware Introspection (GPU vs CPU)

TextLens automatically inspects system hardware using PyTorch upon initialization. Developers can also query hardware capabilities programmatically or print a clean summary:

```python
import textlens

# Print hardware & device summary to terminal
textlens.print_hardware_status()

# Or retrieve structured hardware object
hw = textlens.get_hardware_info()
print(f"GPU Available : {hw.gpu_available}")
print(f"Active Device : {hw.device_type}")
print(f"GPU Name      : {hw.gpu_name}")
print(f"Total VRAM    : {hw.vram_total_gb} GB")
```

**Console Output:**
```text
============================================================
           TEXTLENS HARDWARE & DEVICE STATUS           
============================================================
 PyTorch Version : 2.2.0
 CUDA Compiled   : 12.1
 CPU Cores       : 16
 CUDA (GPU)      : ✅ AVAILABLE (100% Accelerated)
 GPU Model       : NVIDIA GeForce RTX 3080
 Total VRAM      : 10.0 GB
 VRAM Allocated  : 0.45 GB
 Active Target   : CUDA (Default model target: 'cuda')
============================================================
```

---

## 💡 Quickstart: Python SDK

### 1. Basic Text Extraction

```python
from textlens import TextLens

# Auto-detects GPU (CUDA) or CPU
ocr = TextLens()

# Extract text from local image or remote URL
text = ocr.read("sample_receipt.jpg")
print("Extracted Text:\n", text)
```

### 2. Multi-Page PDF Reading

```python
# Render and extract text from every page of a PDF
pages = ocr.read_pdf("financial_report.pdf")

for item in pages:
    print(f"\n--- Page {item['page']} of {item['total_pages']} ---")
    print(item["text"])
```

### 3. Specialized Extractions (Tables, Formulas, JSON)

```python
# Extract table as Markdown
markdown_table = ocr.extract_table("data_table.png")

# Extract mathematical equations as LaTeX
latex_formula = ocr.extract_formula("math_problem.jpg")

# Extract document fields as structured JSON
json_output = ocr.extract_json(
    "invoice.pdf",
    schema='{"invoice_number": "string", "total_amount": "float", "date": "string"}'
)
```

### 4. Dynamic Runtime Device Switching (CUDA ↔ CPU)

```python
# Switch model to CPU on demand
ocr.switch_device("cpu")
text_cpu = ocr.read("document.png")

# Switch back to GPU (CUDA)
ocr.switch_device("cuda")
text_gpu = ocr.read("document.png")
```

---

## 🌐 One-Line REST API Server Hosting

Developers can host a production-ready REST API endpoint in **one line of Python** or via the CLI:

### Option A: Python Code
```python
import textlens

# Launch FastAPI REST server on port 8000
textlens.serve(host="0.0.0.0", port=8000)
```

### Option B: Command Line Interface (CLI)
```bash
textlens serve --port 8000
```

Once running:
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **OCR REST Endpoint**: `http://localhost:8000/api/v1/ocr`
- **Hardware Info Endpoint**: `http://localhost:8000/api/v1/hardware`

---

## 📡 REST API Endpoint Integration Guide

Any application (web, mobile, backend server, scripts) can send requests to the served URL.

### 1. cURL Example (File Upload)

```bash
curl -X POST "http://localhost:8000/api/v1/ocr" \
  -F "file=@/path/to/invoice.png" \
  -F "prompt=Text Recognition:"
```

### 2. Python (`requests`) Example

```python
import requests

# File upload request
url = "http://localhost:8000/api/v1/ocr"
files = {"file": open("invoice.png", "rb")}
data = {"prompt": "Text Recognition:"}

response = requests.post(url, files=files, data=data)
result = response.json()

print("Status:", result["status"])
print("Device Used:", result["device_used"])
print("Text:", result["text"])
```

### 3. JavaScript (`fetch`) Example

```javascript
const formData = new FormData();
formData.append("file", fileInputElement.files[0]);
formData.append("prompt", "Text Recognition:");

fetch("http://localhost:8000/api/v1/ocr", {
  method: "POST",
  body: formData
})
  .then(res => res.json())
  .then(data => {
    console.log("Extracted Text:", data.text);
    console.log("Execution Time (s):", data.execution_time_seconds);
  });
```

---

## 🖥️ Command Line Interface (CLI) Usage

```bash
# Print system GPU/CPU hardware diagnostic status
textlens hardware

# Run OCR directly on a local file or URL
textlens read document.jpg

# Run PDF OCR with custom prompt
textlens read report.pdf --prompt "Extract table content:"

# Launch REST API server
textlens serve --host 0.0.0.0 --port 8000
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
