# PDFs and selective OCR

Never OCR a whole PDF blindly. TextLens inspects every page first and only
renders and OCRs the pages that need it.

```text
PDF ─► inspect (ms/page, no rendering) ─┬─ native text ─► positioned extraction ─► quality check
                                         ├─ scanner OCR layer ─► reuse (fast/edge) or re-OCR
                                         ├─ blank ─► skip
                                         └─ needs OCR ─► render ─► route ─► OCR ─► validate
```

## Inspect first

```bash
textlens inspect report.pdf
textlens inspect huge.pdf --sample 8      # triage a 2,000-page file in milliseconds
textlens inspect report.pdf --json
```

```python
report = OCR().inspect("report.pdf")
report.pdf_type            # text_based | scanned | image_based | mixed | empty
report.pages_needing_ocr   # [3, 4, 17]
report.reasons_by_page     # {3: ["scanned"], 17: ["suspected_garbled_text"]}
```

### Page kinds and reason codes

| Reason | Meaning | Action |
|---|---|---|
| `scanned` | no visible text; an image covers the page | OCR |
| `invisible_text_layer` | a scanner's hidden OCR text sits under a page image | reuse (fast/edge) or re-OCR (balanced+) |
| `suspected_garbled_text` | text layer exists but decodes to garbage (broken ToUnicode, CID fonts, cipher-shifted letters, U+FFFD runs) | OCR |
| `vector_text` | glyphs drawn as vector outlines; no text operators | OCR |
| `sparse_text_over_image` | a few words (stamp, header) over a full-page scan | OCR |
| `no_text` | a few unusable glyphs, nothing trustworthy | OCR |

Pages with no text and nothing drawn are `empty` and skipped. Native pages
with large embedded images are `mixed`; pass `ocr_images=True` to also OCR
text inside those images (screenshots, figures) — the page becomes `fused`.

The inspector also flags complexity for routing: ruling lines (`likely_table`)
and math fonts/characters (`likely_formulas`).

These codes follow Firecrawl pdf-inspector's vocabulary; see
[the design note](../design/pdf-inspector.md).

## Native extraction

For native pages, TextLens reads characters with their boxes from PDFium and
rebuilds words → lines → paragraphs in reading order (multi-column aware),
types headings by font size, detects list items and page furniture, and
reconstructs ruled tables. Coordinates are PDF points with a top-left origin,
rotation applied.

Before trusting a text layer, TextLens runs its quality checks; a garbled
layer is sent to OCR instead.

## OCR modes

```python
OCR(ocr="auto")    # selective (default)
OCR(ocr="force")   # OCR every page (e.g. you distrust all text layers)
OCR(ocr="off")     # native text only; scanned pages come back empty
OCR(reuse_ocr_layer=False)   # always re-OCR scanner OCR layers
```

```bash
textlens ocr doc.pdf --ocr force --pages 1-5 --dpi 300
```

## Large documents

Processing is page-by-page and streaming: memory stays flat regardless of
page count.

```python
for page in OCR().stream("archive.pdf"):
    index(page.number, page.text)

doc = textlens.load("archive.pdf")   # lazy
doc.page_count                        # cheap
doc.page(812).text                    # processes one page
doc.inspect()                         # classification only
```

- **Page selection**: `pages=[1, 2, 10]`, CLI `--pages 1-3,10-`.
- **Limits**: documents above `TEXTLENS_MAX_PAGES` (default 5000) are refused
  unless you select pages.
- **Partial results**: a page that fails is returned empty with
  `needs_review=True` and a warning; the rest of the document completes.
- **Resume**: `textlens batch --resume` skips files already exported; for a
  single huge file, stream pages and checkpoint as you go.
- **Encrypted PDFs**: `OCR(password="…")` or `--password`.

## Why selective OCR matters

On a mixed PDF, every native page skipped saves rendering (≈50–200 ms) and
inference (≈0.3–1 s on CPU for PP-OCR, seconds for a VLM), avoids
recognition errors on text that was already exact, and keeps GPU capacity for
the pages that need it. Measure on your own corpus:

```bash
textlens profile report.pdf            # time per stage, pages by source
textlens ocr report.pdf --ocr force    # compare against OCR-everything
```
