# Results

Every call returns a `Result` with the same shape, whichever engine ran.

```text
Result
 ├── pages[]                 Page(number, width, height, unit, classification)
 │    ├── blocks[]           Block(type, text, bbox, confidence, source, model, backend, order, id, level)
 │    │    └── lines[]       Line(text, bbox, confidence, polygon, font_size, bold)
 │    │         └── words[]  Word(text, bbox, confidence)
 │    ├── tables[]           Table(cells[], bbox, confidence) → grid(), to_markdown(), to_csv(), to_html()
 │    ├── formulas[]         Formula(latex, bbox, confidence)
 │    ├── markdown           engine-native Markdown (generative models)
 │    └── provenance         PageProvenance (see below)
 ├── metadata                source name, mime, size, page count, PDF title/author…
 └── provenance             RunProvenance: version, profile, document hash, config hash, models, timing
```

## Convenience views

```python
result.text          # all text in reading order
result.blocks        # every block, page by page
result.lines, result.words, result.tables, result.formulas
result.layout        # [{page, type, bbox, order, id, source}]
result.confidence    # length-weighted mean (native text = 1.0)
result.pages_needing_review()
```

## Coordinates

`bbox = (x0, y0, x1, y1)` with a **top-left origin**, in the page's unit:

| Input | `page.unit` | meaning |
|---|---|---|
| PDF page | `pt` | PDF points (1/72 inch), `/Rotate` and CropBox already applied |
| image | `px` | pixels of the original image |
| DOCX | `flow` | no geometry (boxes are `None`) |

To draw a PDF box on a page rendered at `dpi`, multiply by `dpi / 72`.
Generative models (VLMs) produce text without positions, so their blocks
have `bbox=None`.

## Block types

`title`, `heading` (with `level`), `text`, `list_item`, `table`, `formula`,
`figure`, `caption`, `code`, `page_header`, `page_footer`, `footnote`.
Running headers/footers repeated across pages are detected and excluded from
Markdown and chunks by default.

## Provenance

Provenance answers "where did this text come from, and can I trust it?".

```python
p = result.pages[1].provenance
p.source          # native | ocr | vlm | fused | empty
p.model, p.model_revision, p.backend, p.device
p.dpi             # render resolution if the page was OCR'd
p.reasons         # why OCR ran: scanned, invisible_text_layer, suspected_garbled_text…
p.routing         # the router's explanation
p.attempts        # every model tried: confidence, accepted?, reason, time
p.timings_ms      # inspect, extract, render, detect, recognize, generate…
p.warnings        # e.g. "reused the document's existing OCR text layer"
p.needs_review    # confidence below the profile's review threshold
```

`result.provenance` is the pipeline fingerprint: TextLens version, schema
version, profile, `document_hash` (SHA-256 of the input), `config_hash`
(every option that affects output), models, device and timing. Store it next
to results to reproduce or audit them later.

## Confidence

| Source | Meaning |
|---|---|
| native text | 1.0 (exact text layer, after quality checks) |
| PP-OCR | mean per-character probability of each line |
| VLM (local) | not calibrated → `None` |
| VLM (remote with logprobs) | mean token probability (a proxy) |

Confidence scales differ between engines; use them per engine, and check
`textlens benchmark`'s `confidence_gap` to see whether confidence predicts
errors on your data.

## Exports

```python
result.to_text()
result.to_markdown(page_markers=True)   # "<!-- page: N -->" markers for citations
result.to_html()                        # <section data-page>, data-bbox on every block
result.to_json() / result.to_dict()     # lossless; Result.from_json() round-trips
result.to_csv()                         # all tables
result.to_chunks(max_tokens=500)        # RAG, see tasks/rag.md
result.save("out/", formats=("json", "md", "txt", "html"))
```

`save()` writes `document.json`, `document.md`, `document.txt`,
`document.html`, `tables/page-N-table-K.csv` and `provenance.json`.
