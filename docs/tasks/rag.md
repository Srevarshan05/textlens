# OCR for RAG

Retrieval systems need text that is clean, structured and **citable**. TextLens
chunks keep the pages and boxes each chunk came from.

```python
from textlens import OCR

result = OCR()("handbook.pdf")
for chunk in result.to_chunks(max_tokens=400, overlap_tokens=40, strategy="semantic"):
    store.add(
        text=chunk.text,
        metadata={
            "doc": chunk.document_id,       # SHA-256 of the file
            "pages": chunk.pages,           # [12, 13]
            "section": " > ".join(chunk.section),   # "Benefits > Parental leave"
            "boxes": chunk.bboxes,          # [{"page": 12, "bbox": [x0, y0, x1, y1]}, …]
            "blocks": chunk.block_ids,
        },
    )
```

## Strategies

| `strategy` | Behaviour |
|---|---|
| `semantic` (default) | a new chunk at every heading; each chunk records its heading path; no overlap across sections |
| `block` | packs blocks greedily up to `max_tokens` |
| `page` | one chunk per page (split further if too long) |

Blocks longer than `max_tokens` split at sentence boundaries. Page
headers/footers are excluded unless `include_furniture=True`. Token counts
use a 4-characters-per-token estimate; pass `tokenizer=len_fn` for exact
counts with your embedding model's tokenizer.

## Markdown with page markers

```python
md = result.to_markdown(page_markers=True)   # "<!-- page: 12 -->" before each page
```

Useful when your pipeline chunks Markdown itself but still needs citations.

## Citations back to the source

Chunk boxes are in PDF points (top-left origin). To highlight a cited span on
a page rendered at `dpi`, multiply by `dpi / 72`. The HTML export carries
`data-page` and `data-bbox` on every element.

## Ingestion at scale

- Use `OCR().batch(folder, workers=N)` or the server's `/batch` endpoint.
- Cache by content: identical files are recognised by SHA-256
  (`TEXTLENS_RESULT_CACHE=disk` to persist across runs).
- Keep `result.provenance` (model, config hash) with your chunks so you can
  re-index only what a model upgrade changes.
