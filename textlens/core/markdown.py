"""
textlens.core.markdown
──────────────────────
Parse Markdown (as emitted by document VLMs such as GLM-OCR, LightOnOCR,
HunyuanOCR or PaddleOCR-VL) into TextLens blocks, tables and formulas.

VLMs return a single string; without this step their output would be an
opaque blob.  The parser recognises ATX headings, pipe tables, HTML tables,
``$$…$$`` / ``\\[…\\]`` display math, fenced code, list items and
paragraphs.  Boxes are unknown for generative output, so ``bbox`` is left as
``None`` and provenance marks the blocks as ``vlm``.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Optional, Tuple

from textlens.core.result import (
    CODE,
    FORMULA,
    HEADING,
    LIST_ITEM,
    SOURCE_VLM,
    TABLE,
    TEXT,
    TITLE,
    Block,
    Formula,
    Line,
    Table,
    TableCell,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST = re.compile(r"^\s*([-*+•]|\d+[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cells: List[TableCell] = []
        self._row = -1
        self._col = 0
        self._in_cell = False
        self._buf: List[str] = []
        self._span = (1, 1)
        self._header = False
        self._occupied: set = set()

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._row += 1
            self._col = 0
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._in_cell = True
            self._buf = []
            self._header = tag == "th"
            self._span = (_int(a.get("rowspan")), _int(a.get("colspan")))
        elif tag == "br" and self._in_cell:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_cell:
            row = max(self._row, 0)
            while (row, self._col) in self._occupied:
                self._col += 1
            rs, cs = self._span
            for r in range(row, row + rs):
                for c in range(self._col, self._col + cs):
                    self._occupied.add((r, c))
            self.cells.append(
                TableCell(
                    row=row,
                    col=self._col,
                    text=" ".join("".join(self._buf).split()),
                    row_span=rs,
                    col_span=cs,
                    is_header=self._header or row == 0,
                )
            )
            self._col += cs
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._buf.append(data)


def _int(value: Optional[str]) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def parse_html_table(fragment: str) -> Optional[Table]:
    parser = _HTMLTableParser()
    try:
        parser.feed(fragment)
    except Exception:  # malformed model output must never crash parsing
        return None
    return Table(cells=parser.cells) if parser.cells else None


def _split_row(line: str) -> List[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells = re.split(r"(?<!\\)\|", body)
    return [c.replace("\\|", "|").strip() for c in cells]


def parse_markdown(
    markdown: str,
    page: int = 1,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    confidence: Optional[float] = None,
    source: str = SOURCE_VLM,
) -> Tuple[List[Block], List[Table], List[Formula]]:
    """Convert Markdown into ``(blocks, tables, formulas)``."""
    blocks: List[Block] = []
    tables: List[Table] = []
    formulas: List[Formula] = []
    lines = (markdown or "").replace("\r\n", "\n").split("\n")
    i = 0
    para: List[str] = []

    def add(block_type: str, text: str, **attrs: object) -> Block:
        blk = Block(
            type=block_type,
            text=text,
            confidence=confidence,
            page=page,
            source=source,
            model=model,
            backend=backend,
            lines=[Line(t, None, confidence) for t in text.split("\n") if t.strip()],
            order=len(blocks),
            id=f"p{page}-b{len(blocks)}",
        )
        level = attrs.pop("level", None)
        if level is not None:
            blk.level = int(level)  # type: ignore[arg-type]
        blk.attributes.update(attrs)
        blocks.append(blk)
        return blk

    def flush_para() -> None:
        if para:
            text = "\n".join(para).strip()
            if text:
                add(TEXT, text)
            para.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_para()
            i += 1
            continue

        # Fenced code
        if stripped.startswith("```"):
            flush_para()
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                body.append(lines[j])
                j += 1
            add(CODE, "\n".join(body))
            i = j + 1
            continue

        # Display math: $$ … $$ (single or multi-line) and \[ … \]
        if stripped.startswith("$$") or stripped.startswith("\\["):
            flush_para()
            opener, closer = ("$$", "$$") if stripped.startswith("$$") else ("\\[", "\\]")
            inner = stripped[len(opener):]
            body = []
            if inner.endswith(closer) and len(inner) >= len(closer):
                body.append(inner[: -len(closer)])
                j = i
            else:
                if inner:
                    body.append(inner)
                j = i + 1
                while j < len(lines) and closer not in lines[j]:
                    body.append(lines[j])
                    j += 1
                if j < len(lines):
                    body.append(lines[j].split(closer)[0])
            latex = "\n".join(body).strip()
            if latex:
                formulas.append(Formula(latex=latex, page=page, confidence=confidence, source=source, model=model))
                add(FORMULA, latex, latex=latex, formula_index=len(formulas) - 1)
            i = j + 1
            continue

        # HTML tables (HunyuanOCR, PaddleOCR-VL, GLM-OCR table mode)
        if stripped.lower().startswith("<table"):
            flush_para()
            j = i
            buf = []
            while j < len(lines):
                buf.append(lines[j])
                if "</table>" in lines[j].lower():
                    break
                j += 1
            table = parse_html_table("\n".join(buf))
            if table is not None:
                table.page, table.model, table.source, table.confidence = page, model, source, confidence
                tables.append(table)
                add(TABLE, table.text, table_index=len(tables) - 1)
            i = j + 1
            continue

        # Pipe tables
        if _TABLE_ROW.match(line) and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush_para()
            rows = [_split_row(line)]
            j = i + 2
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                rows.append(_split_row(lines[j]))
                j += 1
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            table = Table.from_grid(rows, header=True, page=page, model=model, source=source, confidence=confidence)
            tables.append(table)
            add(TABLE, table.text, table_index=len(tables) - 1)
            i = j
            continue

        # Headings
        m = _HEADING.match(stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            add(TITLE if level == 1 else HEADING, m.group(2).strip(), level=level)
            i += 1
            continue

        # List items
        if _LIST.match(line):
            flush_para()
            add(LIST_ITEM, stripped)
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_para()
    return blocks, tables, formulas
