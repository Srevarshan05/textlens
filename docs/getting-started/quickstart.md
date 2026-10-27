# Quickstart

## One line

```python
from textlens import OCR

print(OCR()("invoice.pdf").text)
```

`OCR()` inspects the input, extracts native PDF text where it exists, OCRs
only what needs it, and picks a model for your hardware.

## The result

```python
result = OCR()("report.pdf")

result.text              # plain text, reading order
result.to_markdown()     # headings, lists, tables, formulas
result.to_json()         # full schema: pages → blocks → lines → words
result.confidence        # 0–1 (native text counts as 1.0)

for page in result.pages:
    print(page.number, page.provenance.source, page.provenance.model)
    for block in page.blocks:
        print(block.type, block.bbox, block.text[:40])
```

Every block knows its page, bounding box, confidence, source (`native`,
`ocr`, `vlm`, `fused`) and model. See [Results](../concepts/results.md).

## The CLI

```bash
textlens ocr receipt.jpg                      # text to stdout
textlens ocr report.pdf -f markdown -o report.md
textlens ocr scan.pdf --pages 1-3 --explain   # show routing per page
textlens inspect big.pdf                      # which pages need OCR, and why (no OCR)
textlens document paper.pdf                   # structure-first Markdown
textlens extract invoice.pdf -s "invoice_number,date,total"
```

## Choosing speed vs accuracy

```python
OCR(profile="edge")      # smallest footprint (Raspberry Pi)
OCR(profile="fast")      # PP-OCR on CPU/GPU
OCR(profile="balanced")  # fast engine for simple pages, a VLM for hard ones
OCR(profile="accurate")  # best installed model first
OCR(model="glm-ocr")     # pin a model
```

Without a profile, `auto` picks one from your hardware. `textlens setup`
saves your choice. See [Routing](../concepts/routing.md).

## Large documents

```python
ocr = OCR()
for page in ocr.stream("book.pdf"):          # constant memory
    print(page.number, len(page.text))

doc = textlens.load("book.pdf")              # lazy document
doc.page(120).text                           # processes only page 120
```

## Many files

```python
results = OCR().batch("./invoices/", workers=4)   # failures are returned, not raised
```

```bash
textlens batch ./invoices -o results/ --resume   # live dashboard at :8765; --no-dashboard to skip
```

## A REST API

```bash
pip install "textlens-ocr[server]"
textlens serve --port 8000
curl -F file=@scan.pdf localhost:8000/ocr
```

See [Server](../deployment/server.md).

## Next steps

- [PDFs and selective OCR](../tasks/pdf.md)
- [RAG chunks with citations](../tasks/rag.md)
- [Benchmark models on your own data](../evaluation/benchmarking.md)
