# Structured extraction

```python
ext = OCR().extract("invoice.pdf", schema={
    "vendor": "string",
    "invoice_number": "string",
    "date": "date",
    "total": "float",
})
ext.data      # {"vendor": "...", "invoice_number": "INV-42", "date": "2026-03-04", "total": 1234.5}
ext.fields    # per field: value, raw text, label, page, bbox, confidence, method
```

```bash
textlens extract invoice.pdf -s "invoice_number,date,total"
textlens extract invoice.pdf -s schema.json --details
```

## Schemas

- a mapping: `{"total": "float", "date": "date"}` — types: `string`, `number`/`float`, `integer`, `date`, `boolean`
- a list of names: `["invoice_number", "total"]` — types are inferred from
  names (`total`, `amount`, `tax` → number; `*_date`, `date` → date)
- a JSON Schema object (`{"type": "object", "properties": {...}}`)
- a Pydantic model class

Dates are normalised to ISO `YYYY-MM-DD`; amounts handle `$1,234.50`,
`1.234,50 €` and `(12.00)`.

## Strategies

| Strategy | How | When |
|---|---|---|
| `vlm` | a document VLM (local or served) answers with JSON for the schema | messy layouts, implicit fields |
| `heuristic` | local rules over the OCR layout: `Label: value`, label then value on the next line, `LABEL 12.00`, table rows and headers, typed date patterns, synonyms (`total` ≈ `amount due`, `grand total`…) and fuzzy label matching | fully offline, no GPU, auditable |
| `auto` (default) | `vlm` if a generative model can run, else `heuristic` | |

Heuristic results keep the page and bounding box of each value, so you can
highlight it for human review. Always validate extracted values that drive
payments or decisions.
