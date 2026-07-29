<p align="center">
  <img src="Text-Lens.png" alt="TextLens Logo" width="380" />
</p>

<h1 align="center">TextLens</h1>

<p align="center">
  <strong>Next-Generation Multimodal Visual Document Intelligence & Structured Data Extraction SDK</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/textlens/"><img src="https://img.shields.io/badge/PyPI-v0.1.0-blue.svg?style=for-the-badge&logo=pypi" alt="PyPI Package"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"></a>
  <a href="https://huggingface.co/zai-org/GLM-OCR"><img src="https://img.shields.io/badge/Model-GLM--OCR-yellow.svg?style=for-the-badge&logo=huggingface" alt="GLM-OCR Model"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License"></a>
  <a href="#-contributing--community"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge&logo=github" alt="PRs Welcome"></a>
</p>

<p align="center">
  <a href="#-what-is-textlens">What is TextLens?</a> •
  <a href="#-key-features--capabilities">Key Features</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-advanced-usage-examples">Usage Examples</a> •
  <a href="#-cli-interface">CLI Interface</a> •
  <a href="#-fastapi-server-mode">FastAPI Server</a> •
  <a href="#-contributing--community">Contributing</a> •
  <a href="#-status--roadmap">Roadmap</a>
</p>

---

## 💡 What is TextLens?

**TextLens** is an open-source, high-performance Python SDK and microservice engine for **Multimodal Document Understanding**. Built atop the cutting-edge **GLM-OCR** vision-language architecture by Z.ai, TextLens bridges the gap between raw document images and structured down-stream AI pipelines.

Rather than relying on legacy bounding-box OCR heuristics or fragile regex parsing, TextLens leverages autoregressive vision-language modeling to perform **zero-shot document parsing**, extracting natural text, dense Markdown tables, complex LaTeX formulas, and schema-constrained JSON fields in a single inferencing pass.

Whether running locally on consumer hardware, accelerating via CUDA on enterprise GPUs, or deploying as a cloud-native HTTP microservice, TextLens delivers production-grade document intelligence out of the box.

---

## ✨ Key Features & Capabilities

- 👁️ **Multimodal OCR & Layout Understanding**: Parses dense text, multi-column articles, historical scans, and multi-page PDFs while preserving spatial layout and reading order.
- 📊 **Latent Table Extraction**: Automatically transcribes nested visual tables directly into clean GitHub-Flavored Markdown or structured pandas-compatible dicts.
- 🧮 **LaTeX Equation Recognition**: Converts inline mathematical expressions and multi-line equations into precise LaTeX notation.
- 🎯 **Zero-Shot Schema Extraction**: Enforce custom Pydantic schemas or JSON templates to extract target key-value pairs (e.g., invoices, IDs, medical forms, receipts).
- ⚡ **Heterogeneous Acceleration**: Auto-detects device hardware — seamlessly routes compute across NVIDIA GPUs (CUDA), Apple Silicon (MPS), or optimized CPU runtimes.
- 🚀 **High-Throughput Batch Processing**: Asynchronous worker pools and batched tensor pipelines for bulk ingestion of file directories.
- 🌐 **Instant FastAPI Server Mode**: Spin up a production-ready HTTP REST API for remote document parsing with a single CLI command or Python call.
- 🛠️ **Developer-First CLI**: Robust command-line interface for rapid shell scripts, CI/CD pipelines, and local benchmarking.

---

## ⚙️ Ingestion & Parsing Architecture

```text
┌───────────────────────────┐      ┌──────────────────────────────┐
│ Input Asset               │      │ Heterogeneous Compute Engine │
│ (Image / PDF / Directory) ├─────►│  Auto-Device Allocation      │
└───────────────────────────┘      │  [CUDA / MPS / CPU]          │
                                   └──────────────┬───────────────┘
                                                  │
                                                  ▼
┌───────────────────────────┐      ┌──────────────────────────────┐
│ Output Formatter Engine   │      │ GLM-OCR Vision-Language VLM  │
│ [Markdown / LaTeX / JSON] ◄──────┤ Autoregressive Layout Parser │
└───────────────────────────┘      └──────────────────────────────┘
```

---

## 📦 Installation

### Standard PyPI Install
```bash
pip install textlens
```

### GPU / CUDA Accelerated Install
```bash
pip install textlens[cuda]
```

### Developer / Source Install
```bash
git clone https://github.com/Srevarshan05/textlens.git
cd textlens
pip install -e .[dev]
```

---

## 🚀 Quickstart

```python
from textlens import TextLens

# Initialize engine (Auto-detects GPU / CPU hardware acceleration)
ocr = TextLens()

# Read raw text from any document image
text = ocr.read("invoice.png")
print(text)

# Process a multi-page PDF document page-by-page
pages = ocr.read_pdf("annual_report.pdf")
for page_num, page_content in enumerate(pages):
    print(f"--- Page {page_num + 1} ---")
    print(page_content)

# Extract tabular data directly formatted as Markdown
table_md = ocr.extract_table("balance_sheet.jpg")
print(table_md)
```

---

## 💻 Advanced Usage Examples

### 1. Schema-Guided Structured Extraction (JSON / Pydantic)
Extract structured fields directly into typed formats according to a custom JSON schema or Pydantic model:

```python
from pydantic import BaseModel
from textlens import TextLens

class InvoiceSchema(BaseModel):
    invoice_number: str
    vendor_name: str
    total_amount: float
    due_date: str

ocr = TextLens()

# Extract structured payload enforcing Pydantic validation
invoice_data = ocr.extract_structured(
    image="receipt.jpg",
    schema=InvoiceSchema
)

print(invoice_data.invoice_number)
print(f"Total: ${invoice_data.total_amount}")
```

### 2. LaTeX Mathematical Formula Extraction
Extract mathematical expressions and scientific equations from papers or textbooks:

```python
from textlens import TextLens

ocr = TextLens()
formulas = ocr.extract_formulas("physics_paper.png")

for idx, formula in enumerate(formulas):
    print(f"Equation {idx + 1}:")
    print(f"$$ {formula} $$")
```

### 3. Asynchronous Batch Processing
Process large volumes of documents concurrently using async pipelines:

```python
import asyncio
from textlens import AsyncTextLens

async def process_batch():
    async_ocr = AsyncTextLens(batch_size=8)
    
    results = await async_ocr.read_batch([
        "docs/doc1.png",
        "docs/doc2.png",
        "docs/doc3.pdf",
    ])
    
    for path, text in results.items():
        print(f"File: {path} | Processed Length: {len(text)} chars")

asyncio.run(process_batch())
```

---

## 🌐 FastAPI Server Mode

Deploy TextLens as an enterprise-grade REST microservice HTTP endpoint in one line:

### Launch via Python
```python
from textlens.server import serve

# Start FastAPI server on port 8000
serve(host="0.0.0.0", port=8000, workers=4)
```

### Launch via CLI
```bash
textlens serve --host 0.0.0.0 --port 8000 --workers 4
```

### OpenAPI Endpoint Benchmark
Once running, interactive API docs are available at `http://localhost:8000/docs`:
- `POST /v1/ocr/read` - Upload image/PDF and receive parsed text.
- `POST /v1/ocr/extract-table` - Extract visual tables as Markdown/JSON.
- `POST /v1/ocr/extract-schema` - Submit image + JSON Schema for structured output.

---

## 🛠️ CLI Interface

TextLens ships with a zero-configuration command-line utility for instant shell pipelines:

```bash
# Process a single image
textlens read invoice.png --format markdown

# Extract tables from a document
textlens extract-table data.jpg --out table.md

# Batch process an entire directory of PDFs & images
textlens batch ./input_docs/ --output-dir ./parsed_results/ --concurrency 4

# Start local API server
textlens serve --port 8000
```

---

## 🤝 Contributing & Community

**TextLens is a community-driven open-source project!** We warmly welcome contributions, architectural ideas, feature suggestions, bug reports, and pull requests from developers and AI researchers worldwide.

### How to Contribute Ideas & Improvements:

1. 💡 **Propose Ideas & Features**: Have an idea for a feature or model optimization? Open a topic in **GitHub Discussions** or submit a **Feature Request Issue**.
2. 🐛 **Report Bugs & Edge Cases**: Found an issue with visual layout parsing or GPU memory management? Submit a detailed bug report with reproducible code snippets.
3. 🔀 **Submit Pull Requests**:
   - Fork the repository.
   - Create your feature branch (`git checkout -b feature/awesome-feature`).
   - Run tests and linters:
     ```bash
     pytest tests/
     ruff check .
     black --check .
     ```
   - Commit your changes (`git commit -m 'feat: Add support for ONNX inference backend'`).
   - Push to the branch (`git push origin feature/awesome-feature`).
   - Open a Pull Request on GitHub.

---

## 🚧 Status & Roadmap

> 🚧 **Status**: Active Development (`v0.1.0`). Core SDK features, CLI tool, and microservice server engine are rapidly evolving.

- [x] Core GLM-OCR model wrapper & auto-device allocation (CUDA/MPS/CPU)
- [x] Multi-page PDF parsing and layout preservation
- [x] Markdown table & LaTeX equation extraction modules
- [x] Built-in FastAPI microservice server module & CLI binary
- [ ] **vLLM & TensorRT-LLM Inference Engine**: Ultra-low latency serving for enterprise workloads
- [ ] **ONNX & INT8 Quantization**: Optimized footprint for edge devices & browser runtimes
- [ ] **Streaming Token Output**: Real-time WebSocket / SSE text streaming for LLM agents
- [ ] **Bounding Box & Layout Inspector**: Interactive visual debugging tool for extracted elements

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments & Credits

- Special thanks to **Z.ai** for training and open-sourcing the powerful [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) model architecture.
- Powered by [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/), and [FastAPI](https://fastapi.tiangolo.com/).

---

<p align="center">
  <sub>Built with ❤️ by open-source contributors for the global AI developer community.</sub>
</p>
