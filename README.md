# textlens

A Python SDK that reads text, tables, formulas, and structured data from images and PDFs using the GLM-OCR model — with batch processing, a FastAPI server mode, and a built-in CLI. Works on GPU or CPU.

---

## What is textlens?

`textlens` wraps the [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) model — a state-of-the-art document understanding model by Z.ai — into a clean, reusable Python SDK that any developer can drop into any project.

It goes beyond plain text extraction:
- Read raw text from any image, scanned document, or PDF
- Extract tables as formatted Markdown
- Extract mathematical formulas
- Pull structured JSON fields from any document using your own schema
- Process entire folders of files in one call
- Serve everything as a FastAPI HTTP endpoint with a single line

---

## Quickstart

```bash
pip install textlens
```

```python
from textlens import TextLens

ocr = TextLens()                     # Auto-detects GPU or CPU

text = ocr.read("invoice.png")       # Read any image
pages = ocr.read_pdf("report.pdf")   # Read a full PDF page by page
table = ocr.extract_table("data.jpg") # Extract a table as Markdown
```

---

## Status

> 🚧 Package is under active development. Core SDK, CLI, and server module coming soon.

---

## License

MIT
