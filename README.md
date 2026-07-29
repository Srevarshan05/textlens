<p align="center">
  <img src="Text-Lens.png" alt="TextLens Logo" width="380" />
</p>

<h1 align="center">TextLens</h1>

<p align="center">
  <strong>Simple. Reusable. Local-first OCR for Python.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Coming%20Soon-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Backend-GLM--OCR-yellow?style=for-the-badge&logo=huggingface" />
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#vision">Vision</a> •
  <a href="#planned-features">Planned Features</a> •
  <a href="#contributing">Contributing</a>
</p>

---

# About

After experimenting with many OCR libraries and models, one thing became obvious.

**The model isn't the hard part. Reusing OCR code across projects is.**

Loading models, handling PDFs, writing prompts, parsing outputs, extracting tables, processing batches, and converting everything into usable Python objects often means writing the same code again and again.

During my experiments, **GLM-OCR** stood out because of its impressive accuracy and surprisingly fast local inference on my RTX 3060 (12 GB VRAM).

That inspired **TextLens**.

**TextLens** is an upcoming open-source Python package focused on making OCR easy to use, easy to reuse, and easy to integrate into any Python project.

It isn't intended to be just another wrapper around GLM-OCR.

The goal is to provide clean, reusable functions that hide repetitive OCR workflows behind a simple Python API, so developers can spend less time building OCR pipelines and more time building their applications.

---

# Vision

Instead of writing hundreds of lines of OCR boilerplate every time...

```python
from textlens import TextLens

ocr = TextLens()

text = ocr.read("image.png")
table = ocr.extract_table("table.png")
invoice = ocr.read_invoice("invoice.pdf")
pages = ocr.read_pdf("report.pdf")
```

Simple.

Reusable.

Local-first.

---

# Planned Features

- Clean and consistent Python API
- Image and PDF text extraction
- Table extraction
- Invoice and receipt parsing
- Custom structured JSON extraction
- Batch document processing
- Background processing queues
- FastAPI server mode
- Command Line Interface (CLI)
- Easy integration into existing Python applications
- Extensible architecture for future OCR backends

---

# GPU Recommendation

TextLens is being designed around **local OCR workflows**.

While CPU support will be available, having an NVIDIA GPU makes a significant difference in inference speed.

If you already have a modern GPU, you'll get a much smoother experience when processing larger documents or batches.

---

# Community

TextLens is still in its early stages, and I'd love to build it together with the community.

If you've worked with OCR before, I'd really like to hear from you.

### What problems have you faced while implementing OCR?

Examples:

- Repetitive boilerplate code
- PDF handling
- Table extraction
- Invoice parsing
- Batch processing
- Poor developer experience
- Slow pipelines
- Difficult integrations
- Anything else you've encountered

What features would make an OCR package genuinely useful in your own projects?

Open an **Issue**, start a **Discussion**, or submit a **Pull Request**. Every suggestion is welcome.

---

# Current Status

🚧 **TextLens is currently under active development.**

The core architecture, documentation, and examples will be published as development progresses.

If you're interested in contributing or following the project, consider giving the repository a ⭐.

---

# Contributing

Whether you're interested in Python, documentation, testing, APIs, developer experience, or OCR workflows, contributions and ideas are always welcome.

Let's build something that makes OCR easier for everyone.

---

# License

This project will be released under the MIT License.

---

<p align="center">
Made with ❤️ by the open-source community.
</p>