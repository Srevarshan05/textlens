<p align="center">
  <a href="https://github.com/Srevarshan05/textlens">
    <img src="https://raw.githubusercontent.com/Srevarshan05/textlens/main/website/assets/logo.png" alt="TextLens Logo" width="180" style="border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);" />
  </a>
</p>

<h1 align="center">TextLens</h1>

<p align="center">
  <strong>High-performance, local VLM-powered OCR framework for Python & CLI.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/textlens-ocr/"><img src="https://img.shields.io/pypi/v/textlens-ocr.svg?color=70df7f&style=flat-square" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/textlens-ocr/"><img src="https://img.shields.io/pypi/pyversions/textlens-ocr.svg?color=3b82f6&style=flat-square" alt="Python Versions" /></a>
  <a href="https://github.com/Srevarshan05/textlens/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Srevarshan05/textlens.svg?color=a855f7&style=flat-square" alt="License" /></a>
  <a href="https://github.com/Srevarshan05/textlens"><img src="https://img.shields.io/github/stars/Srevarshan05/textlens.svg?color=eab308&style=flat-square" alt="GitHub Stars" /></a>
</p>

---

## 🌟 What is TextLens?

**TextLens** makes local Vision-Language Model (VLM) OCR simple, reusable, and blazing fast. Initialize a unified OCR client, pass any image or PDF document, and receive clean, structured text output. 

TextLens includes **model discovery and local caching**, **automated CUDA hardware diagnostics**, a **FastAPI REST server microservice**, and a **parallel BatchOCR engine featuring a real-time live monitoring dashboard with PDF report exporting**.

> 💡 **GPU Acceleration Note:** TextLens is built for local VLM inference and is optimized for **NVIDIA CUDA GPUs** (RTX 4050/3060/4090, etc.). CPU fallback is supported for light testing. Apple Silicon (MPS) optimization is under active development.

---

## ✨ Unique Features & Highlights

### 1. 📊 Parallel BatchOCR Engine & Live Web Dashboard
- **Parallel Document Processing**: Process entire document folders or file lists in parallel with worker thread pools and automatic retry logic.
- **Live Monitoring Dashboard (`http://127.0.0.1:8765`)**: Zero-dependency local web dashboard featuring real-time SSE progress streaming.
- **Real Hardware Telemetry**: Live CPU usage line graphs, RAM allocation, and NVIDIA GPU/VRAM utilization (no dummy lines or placeholder sparklines).
- **Interactive Job Controls**: Pause, Resume, Cancel, and Retry Failed tasks on the fly, with dynamic worker thread scaling.
- **Resizable Interface Panels**: Click and drag the bottom edge of the Tasks list and Live Streaming Console to expand them to any height.
- **📄 Printable PDF Execution Report Exporter**: Download standard, print-ready PDF reports (`TextLens_BatchOCR_Report.pdf`) summarizing execution metrics, configuration, hardware stats, and per-file task results with one click.

### 2. 🤖 Supported VLM Backends
| Model ID | Model Name | Primary Capabilities | Recommended VRAM |
| :--- | :--- | :--- | :--- |
| `glm-ocr` | **GLM OCR** *(Default)* | General document OCR, tables, formulas, receipts | **6 GB VRAM** |
| `lighton-ocr` | **LightOnOCR** | Academic papers, complex layouts, multilingual docs | **8 GB VRAM** |
| `hunyuan-ocr` | **HunyuanOCR** | Dense document extraction, charts, structured JSON | **8 GB VRAM** |
| `smolvlm` | **SmolVLM-256M** | Fast lightweight OCR for laptops and low-VRAM GPUs | **2 GB VRAM** |

### 3. ⚡ Zero-Bloat Modular Extras
The core `textlens-ocr` package remains ultra-lightweight (**< 1 MB**). Heavier AI libraries (`torch`, `transformers`, `fastapi`, `rich`) load on-demand when feature extras are requested.

---

## 📦 Installation

TextLens requires Python 3.9+.

```bash
# Basic installation (Model catalog & CLI utilities)
pip install textlens-ocr
```

### Choose your feature extras:

| Requirements | Install Command |
| :--- | :--- |
| **Catalog & CLI Utilities** | `pip install textlens-ocr` |
| **Full VLM OCR & BatchOCR Dashboard** | `pip install "textlens-ocr[inference,ui]"` |
| **REST API Server Microservice** | `pip install "textlens-ocr[inference,server]"` |
| **Live Hugging Face Candidate Discovery** | `pip install "textlens-ocr[catalog]"` |
| **All Features (Complete Developer Suite)** | `pip install "textlens-ocr[all]"` |

> 🛠️ **NVIDIA CUDA Setup:** For GPU acceleration, install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> pip install "textlens-ocr[inference,ui]"
> ```

---

## 💻 CLI Commands Reference

TextLens comes with a fast, zero-delay Command Line Interface:

```bash
# 1. Inspect local hardware, CUDA drivers, and VRAM compatibility
textlens doctor

# 2. View official VLM model catalog and download status
textlens models

# 3. Discover live Hugging Face VLM models matching your GPU
textlens discover --compatible

# 4. Perform instant OCR on a single image or PDF document
textlens read invoice.png --model glm-ocr --device cuda

# 5. Run parallel BatchOCR with live monitoring dashboard
textlens batch ./documents/ --model glm-ocr --workers 2 --format json

# 6. Start the local REST API server with Swagger UI
textlens serve --host 127.0.0.1 --port 8000

# 7. Model lifecycle management
textlens model install smolvlm
textlens model info glm-ocr
textlens model remove smolvlm
```

---

## 🐍 Python API Examples

### Modern `OCR` API (Recommended)

```python
from textlens import OCR

# Initialize OCR client (auto-downloads model on first run)
ocr = OCR(model="glm-ocr", device="cuda")

# Extract text from image or PDF
text = ocr.read("invoice.png", dpi=200)
print(text)
```

### Parallel BatchOCR & Dashboard API

```python
from textlens.batch import BatchOCR, TaskStatus

# Create BatchOCR instance
batch = BatchOCR(
    model="glm-ocr",
    workers=2,
    output_dir="./ocr_results",
    output_format="json",
    enable_dashboard=True,   # Launches live web dashboard at http://127.0.0.1:8765
)

# Run batch job
tasks = batch.run("./documents_folder")

for task in tasks:
    if task.status == TaskStatus.COMPLETED:
        print(f"✓ {task.source_path.name} -> {task.output_path}")
    else:
        print(f"✗ {task.source_path.name} -> Error: {task.error}")
```

### Legacy SDK Helpers (`TextLens`)

```python
from textlens import TextLens

engine = TextLens(auto_load=False)
engine.load()

# Structured extraction helpers
print(engine.extract_table("financial_table.png"))
print(engine.extract_formula("math_paper.png"))
print(engine.extract_json("receipt.png", schema='{"store": "str", "total": "float"}'))
```

---

## 🌐 Local REST API Microservice

Launch a local OCR microservice with built-in OpenAPI documentation:

```bash
textlens serve --host 127.0.0.1 --port 8000
```

- **Swagger UI**: Interactive documentation at `http://127.0.0.1:8000/docs`
- **Endpoints**:
  - `GET /api/v1/health` — Service health & version
  - `GET /api/v1/hardware` — GPU VRAM & system metrics
  - `POST /api/v1/ocr` — Document upload OCR endpoint

---

## 🧪 Testing & Verification

Run feature checks and test suites locally:

```bash
python -m pytest tests -q
python scripts/run_feature_checks.py
```

---

## 📜 License

MIT License © TextLens Contributors.
