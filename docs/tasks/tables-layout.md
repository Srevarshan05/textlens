# Tables and layout

## Reading order

TextLens orders text with a recursive XY-cut over word (native) or line (OCR)
boxes: it splits at the widest whitespace gaps — horizontal gaps read top to
bottom, vertical gutters read column by column.

A vertical split is only accepted when **both** sides look like prose columns
(several rows, several words per row). Invoices, forms and key/value sheets
therefore stay row-major: `Design Service ⇥ $00 ⇥ $0.00` rather than all
labels first and all amounts later. Wide gaps inside a line become tabs.

```python
for block in result.pages[0].ordered_blocks():
    print(block.order, block.type, block.bbox)
```

Check it visually: `textlens inspect doc.pdf --overlay out/` numbers every
block.

## Block types

Headings come from font size relative to the page's body text (native PDFs)
or line height (OCR); titles are the largest short text near the top of the
first page; list items from bullets/numbering; page headers/footers from
margins, page-number patterns and repetition across pages.

## Tables

| Source | How tables are found |
|---|---|
| Native PDF | ruling lines (stroked lines and rectangles) → grid → words assigned to cells |
| Document VLMs | Markdown pipe tables and HTML tables (with `rowspan`/`colspan`) parsed from model output |
| PP-OCR | no table structure — text lines only; route table pages to a VLM (`balanced`, `document`) |

```python
for table in result.tables:
    table.page, table.bbox
    table.grid()          # [[...], [...]] dense rows (spans repeated)
    table.to_markdown()
    table.to_csv()
    table.to_html()       # keeps row/col spans
    table.to_records()    # [{header: value}, ...]
```

When a page looks like a table (`likely_table`) and the profile allows
generative models, the router prefers a table-capable model for it.

Borderless tables in native PDFs are currently returned as tab-separated
lines rather than cells; use `task="table"` with a VLM for those pages.

## Formulas

Generative models return LaTeX (`$$…$$` or `\[…\]`), which becomes
`result.formulas` and `formula` blocks. The inspector flags math fonts and
symbols (`likely_formulas`) so `balanced` routes those pages to a
formula-capable model.
