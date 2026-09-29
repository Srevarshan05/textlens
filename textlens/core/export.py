"""
textlens.core.export
────────────────────
Renderers for :class:`~textlens.core.result.Result`: Markdown, HTML,
retrieval chunks and on-disk artifact bundles.

The Markdown output is designed for LLM/RAG consumption: headings become
``#`` levels, tables become pipe tables, formulas become ``$$…$$`` and page
furniture (running headers/footers) is dropped unless requested.  Optional
``<!-- page: N -->`` markers preserve page provenance in plain Markdown.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from textlens.core.result import (
    CAPTION,
    CODE,
    FIGURE,
    FORMULA,
    HEADING,
    LIST_ITEM,
    TABLE,
    TITLE,
    Block,
    Chunk,
    Page,
    Result,
)

# ── Markdown ─────────────────────────────────────────────────────────────────


def _block_markdown(block: Block, page: Page) -> str:
    text = block.text.strip()
    if block.type == TABLE:
        idx = block.attributes.get("table_index")
        if isinstance(idx, int) and 0 <= idx < len(page.tables):
            return page.tables[idx].to_markdown()
        return text
    if block.type == FORMULA:
        latex = block.attributes.get("latex") or text
        return f"$$\n{latex}\n$$" if latex else ""
    if not text:
        if block.type == FIGURE:
            return "<!-- figure -->"
        return ""
    if block.type == TITLE:
        return f"# {text}"
    if block.type == HEADING:
        level = min(max(block.level or 2, 1), 6)
        return f"{'#' * level} {text}"
    if block.type == LIST_ITEM:
        return text if re.match(r"^([-*+•]|\d+[.)])\s", text) else f"- {text}"
    if block.type == CODE:
        return f"```\n{text}\n```"
    if block.type == CAPTION:
        return f"*{text}*"
    return text


def page_to_markdown(page: Page, include_furniture: bool = False) -> str:
    # VLM backends produce Markdown natively; prefer it when the page came
    # straight from such a model and nothing was re-assembled.
    if page.markdown and page.provenance.source == "vlm":
        return page.markdown.strip()
    parts = [_block_markdown(b, page) for b in page.ordered_blocks(include_furniture)]
    rendered_tables = {
        b.attributes.get("table_index") for b in page.blocks if b.type == TABLE and "table_index" in b.attributes
    }
    for i, table in enumerate(page.tables):
        if i not in rendered_tables:
            parts.append(table.to_markdown())
    for f in page.formulas:
        if not any(b.type == FORMULA for b in page.blocks):
            parts.append(f"$$\n{f.latex}\n$$")
    return "\n\n".join(p for p in parts if p)


def result_to_markdown(result: Result, page_markers: bool = False, include_furniture: bool = False) -> str:
    out: List[str] = []
    for page in result.pages:
        md = page_to_markdown(page, include_furniture=include_furniture)
        if page_markers:
            out.append(f"<!-- page: {page.number} -->\n\n{md}" if md else f"<!-- page: {page.number} -->")
        elif md:
            out.append(md)
    return "\n\n".join(out).strip() + "\n" if out else ""


# ── HTML ─────────────────────────────────────────────────────────────────────


def _block_html(block: Block, page: Page) -> str:
    attrs = f' data-page="{page.number}" data-source="{html.escape(block.source)}"'
    if block.bbox is not None:
        attrs += ' data-bbox="' + ",".join(f"{v:.1f}" for v in block.bbox) + '"'
    if block.confidence is not None:
        attrs += f' data-confidence="{block.confidence:.3f}"'
    text = html.escape(block.text.strip())
    if block.type == TABLE:
        idx = block.attributes.get("table_index")
        if isinstance(idx, int) and 0 <= idx < len(page.tables):
            return f"<figure{attrs}>{page.tables[idx].to_html()}</figure>"
    if block.type == TITLE:
        return f"<h1{attrs}>{text}</h1>"
    if block.type == HEADING:
        level = min(max(block.level or 2, 1), 6)
        return f"<h{level}{attrs}>{text}</h{level}>"
    if block.type == LIST_ITEM:
        return f"<li{attrs}>{text}</li>"
    if block.type == FORMULA:
        latex = html.escape(block.attributes.get("latex") or block.text)
        return f'<div class="formula"{attrs}>\\[{latex}\\]</div>'
    if block.type == CODE:
        return f"<pre{attrs}><code>{text}</code></pre>"
    if block.type == CAPTION:
        return f"<figcaption{attrs}>{text}</figcaption>"
    if not text:
        return ""
    return f"<p{attrs}>{text.replace(chr(10), '<br>')}</p>"


def result_to_html(result: Result, full_document: bool = True) -> str:
    sections = []
    for page in result.pages:
        body = "\n".join(filter(None, (_block_html(b, page) for b in page.ordered_blocks(False))))
        if page.provenance.source == "vlm" and not page.blocks and page.markdown:
            body = f"<pre>{html.escape(page.markdown)}</pre>"
        sections.append(
            f'<section class="page" data-page="{page.number}" '
            f'data-width="{page.width:.1f}" data-height="{page.height:.1f}" data-unit="{page.unit}">\n{body}\n</section>'
        )
    content = "\n".join(sections)
    if not full_document:
        return content
    title = html.escape(str(result.metadata.get("title") or result.metadata.get("source_name") or "TextLens result"))
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>{title}</title></head>\n<body>\n{content}\n</body></html>\n"
    )


# ── Chunking for RAG ─────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough, tokenizer-free estimate (≈4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


def chunk_result(
    result: Result,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    strategy: str = "semantic",
    tokenizer: Optional[Callable[[str], int]] = None,
    include_furniture: bool = False,
    document_id: Optional[str] = None,
) -> List[Chunk]:
    """Split a result into citation-preserving chunks.

    Strategies
    ----------
    ``semantic``  pack blocks, starting a new chunk at every heading; each
                  chunk records its heading path (``section``).
    ``block``     pack blocks greedily, ignoring headings.
    ``page``      one chunk per page (split further if a page is too long).

    Every chunk keeps the pages and block boxes it came from, so answers can
    cite and highlight the exact source region.
    """
    if strategy not in ("semantic", "block", "page"):
        raise ValueError("strategy must be 'semantic', 'block' or 'page'")
    count = tokenizer or estimate_tokens
    doc_id = document_id or result.provenance.document_hash
    chunks: List[Chunk] = []
    section: List[str] = []

    class _Acc:
        """Accumulator for the chunk under construction."""

        def __init__(self, carry: str = "", carry_page: Optional[int] = None) -> None:
            self.text: List[str] = [carry] if carry else []
            self.tokens = count(carry) if carry else 0
            self.pages: List[int] = [carry_page] if carry_page is not None else []
            self.boxes: List[Dict[str, Any]] = []
            self.ids: List[str] = []
            self.has_new = False  # True once a block (not just overlap) is added
            self.section: List[str] = list(section)

    acc = _Acc()

    def flush(keep_overlap: bool) -> None:
        nonlocal acc
        if acc.has_new:
            chunks.append(
                Chunk(
                    text="\n\n".join(t for t in acc.text if t).strip(),
                    index=len(chunks),
                    pages=sorted(set(acc.pages)),
                    bboxes=acc.boxes,
                    section=acc.section,
                    block_ids=acc.ids,
                    document_id=doc_id,
                    metadata={"source": result.metadata.get("source_name"), "tokens": acc.tokens},
                )
            )
        carry, carry_page = "", None
        if keep_overlap and overlap_tokens > 0 and acc.has_new and acc.text:
            tail = acc.text[-1]
            limit = overlap_tokens * 4
            carry = tail[-limit:] if len(tail) > limit else tail
            carry_page = acc.pages[-1] if acc.pages else None
        acc = _Acc(carry, carry_page)

    for page in result.pages:
        if strategy == "page":
            flush(keep_overlap=False)
        for block in page.ordered_blocks(include_furniture):
            text = _block_markdown(block, page)
            if not text:
                continue
            if block.type in (TITLE, HEADING) and strategy == "semantic":
                level = 1 if block.type == TITLE else (block.level or 2)
                flush(keep_overlap=False)  # never overlap across sections
                section = section[: max(0, level - 1)] + [block.text.strip()]
                acc.section = list(section)
            if not acc.has_new:
                acc.section = list(section)
            n = count(text)
            # Oversized single blocks are split on sentence boundaries.
            pieces = [text] if n <= max_tokens else _split_long(text, max_tokens, count)
            for piece in pieces:
                if acc.has_new and acc.tokens + count(piece) > max_tokens:
                    flush(keep_overlap=True)
                    acc.section = list(section)
                acc.text.append(piece)
                acc.tokens += count(piece)
                acc.has_new = True
                acc.pages.append(page.number)
                if block.bbox is not None:
                    acc.boxes.append({"page": page.number, "bbox": [round(v, 2) for v in block.bbox]})
                if block.id and block.id not in acc.ids:
                    acc.ids.append(block.id)
    flush(keep_overlap=False)
    return chunks


def _split_long(text: str, max_tokens: int, count: Callable[[str], int]) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    cur = ""
    for s in sentences:
        candidate = f"{cur} {s}".strip()
        if cur and count(candidate) > max_tokens:
            out.append(cur)
            cur = s
        else:
            cur = candidate
    if cur:
        out.append(cur)
    # Hard split anything still too long (e.g. no punctuation).
    final: List[str] = []
    step = max_tokens * 4
    for piece in out:
        if count(piece) <= max_tokens:
            final.append(piece)
        else:
            final.extend(piece[i : i + step] for i in range(0, len(piece), step))
    return final


# ── Artifact bundles ─────────────────────────────────────────────────────────


def save_artifacts(result: Result, directory: Any, formats: Sequence[str] = ("json", "md", "txt")) -> Dict[str, str]:
    """Write a result bundle and return ``{format: path}``.

    Layout::

        <directory>/
            document.json      full result (schema 2.0)
            document.md        RAG-friendly Markdown with page markers
            document.txt       plain text
            document.html      semantic HTML with data-bbox attributes
            tables/page-N-table-K.csv
            provenance.json    run + per-page provenance
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    fmts = {f.lower().lstrip(".") for f in formats}
    if "json" in fmts:
        p = root / "document.json"
        p.write_text(result.to_json(), encoding="utf-8")
        written["json"] = str(p)
    if "md" in fmts or "markdown" in fmts:
        p = root / "document.md"
        p.write_text(result.to_markdown(page_markers=True), encoding="utf-8")
        written["md"] = str(p)
    if "txt" in fmts or "text" in fmts:
        p = root / "document.txt"
        p.write_text(result.text, encoding="utf-8")
        written["txt"] = str(p)
    if "html" in fmts:
        p = root / "document.html"
        p.write_text(result.to_html(), encoding="utf-8")
        written["html"] = str(p)
    if result.tables and ("csv" in fmts or "tables" in fmts or "json" in fmts):
        tdir = root / "tables"
        tdir.mkdir(exist_ok=True)
        for page in result.pages:
            for k, table in enumerate(page.tables, start=1):
                (tdir / f"page-{page.number}-table-{k}.csv").write_text(table.to_csv(), encoding="utf-8")
        written["tables"] = str(tdir)
    prov = {
        "run": result.to_dict()["provenance"],
        "pages": [result.to_dict()["pages"][i]["provenance"] for i in range(len(result.pages))],
    }
    p = root / "provenance.json"
    p.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    written["provenance"] = str(p)
    return written
