<p align="center">
  <img src="https://raw.githubusercontent.com/Srevarshan05/textlens/main/website/assets/logo.png" alt="TextLens" width="140" />
</p>

<h1 align="center">TextLens</h1>

<p align="center"><b>OCR without the OCR complexity.</b><br>
Read text from images and PDFs with one line of Python — on a laptop, a Raspberry Pi or a GPU server.</p>

<p align="center">
  <a href="https://pypi.org/project/textlens-ocr/"><img src="https://img.shields.io/pypi/v/textlens-ocr?color=2f9e44" alt="PyPI"></a>
  <a href="https://pypi.org/project/textlens-ocr/"><img src="https://img.shields.io/pypi/pyversions/textlens-ocr" alt="Python versions"></a>
  <a href="https://github.com/Srevarshan05/textlens/actions/workflows/ci.yml"><img src="https://github.com/Srevarshan05/textlens/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Srevarshan05/textlens/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

<p align="center"><a href="https://textlens-website.vercel.app">Website</a> · <a href="https://github.com/Srevarshan05/textlens/tree/main/docs">Documentation</a> · <a href="https://github.com/Srevarshan05/textlens/tree/main/examples">Examples</a> · <a href="https://github.com/Srevarshan05/textlens/blob/main/CHANGELOG.md">Changelog</a></p>

---

## Install

```bash
pip install textlens-ocr
```

Python 3.10+. No CUDA or PyTorch needed. The first run downloads a small OCR
model (~31 MB) automatically.

## Use it

**1. Get the text**

```python
from textlens import OCR

result = OCR()("invoice.pdf")
print(result.text)
```

**2. Export it**

```python
result.to_markdown()      # headings, lists, tables
result.to_json()          # pages, blocks, bounding boxes, confidence
result.save("output/")    # .json, .md, .txt and tables as .csv
```

**3. Or use the command line**

```bash
textlens ocr invoice.pdf                  # print the text
textlens ocr invoice.pdf -f markdown -o invoice.md
textlens inspect invoice.pdf              # which pages actually need OCR
textlens doctor                           # check your setup
```

That's it. For PDFs, TextLens reads the built-in text of each page directly
and only runs OCR on scanned pages, so it's fast and exact.

## Pick speed or accuracy (optional)

```python
OCR(profile="edge")        # small devices (Raspberry Pi, Jetson)
OCR(profile="fast")        # default on most computers
OCR(profile="accurate")    # use the best model you have installed
OCR(model="glm-ocr")       # choose a specific model
```

Bigger models (for tables, formulas, complex layouts) need a GPU:

```bash
pip install "textlens-ocr[gpu]"
textlens models list
textlens models install glm-ocr
```

## More features

| Feature | Install | Try |
|---|---|---|
| REST API | `pip install "textlens-ocr[server]"` | `textlens serve` |
| Word / PowerPoint files | `pip install "textlens-ocr[documents]"` | `textlens ocr slides.pptx` |
| Number-plate recognition | `pip install "textlens-ocr[anpr]"` | `textlens anpr car.jpg --region in` |
| Extract fields to JSON | — | `textlens extract invoice.pdf -s "invoice_number,date,total"` |
| Many files | — | `textlens batch ./documents -o results/` |
| Docker | — | `docker build -f deploy/docker/Dockerfile -t textlens .` |

## Documentation

- [Quickstart](https://github.com/Srevarshan05/textlens/blob/main/docs/getting-started/quickstart.md) · [Installation](https://github.com/Srevarshan05/textlens/blob/main/docs/getting-started/installation.md) · [Troubleshooting](https://github.com/Srevarshan05/textlens/blob/main/docs/getting-started/troubleshooting.md)
- [How it works](https://github.com/Srevarshan05/textlens/blob/main/docs/concepts/architecture.md) · [Models](https://github.com/Srevarshan05/textlens/blob/main/docs/concepts/models.md) · [API reference](https://github.com/Srevarshan05/textlens/blob/main/docs/API_REFERENCE.md)
- [PDFs](https://github.com/Srevarshan05/textlens/blob/main/docs/tasks/pdf.md) · [RAG](https://github.com/Srevarshan05/textlens/blob/main/docs/tasks/rag.md) · [Edge devices](https://github.com/Srevarshan05/textlens/blob/main/docs/deployment/edge.md) · [Server](https://github.com/Srevarshan05/textlens/blob/main/docs/deployment/server.md) · [Docker](https://github.com/Srevarshan05/textlens/blob/main/docs/deployment/docker.md)
- [Examples](https://github.com/Srevarshan05/textlens/tree/main/examples/) · [Upgrading from 0.x](https://github.com/Srevarshan05/textlens/blob/main/docs/migration.md) · [Changelog](https://github.com/Srevarshan05/textlens/blob/main/CHANGELOG.md)

## Development

```bash
git clone https://github.com/Srevarshan05/textlens.git
cd textlens
pip install -e ".[dev]"
pytest
```

## License

MIT. OCR models are downloaded separately under their own licenses
(`textlens models info <model>`). See [THIRD_PARTY_NOTICES.md](https://github.com/Srevarshan05/textlens/blob/main/THIRD_PARTY_NOTICES.md).
