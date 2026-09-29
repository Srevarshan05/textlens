"""
textlens.documents.pdf.native
─────────────────────────────
Position-aware native text extraction (no OCR).

For a PDF page with a trustworthy text layer this produces the same
:class:`~textlens.core.result.Page` structure an OCR engine would: words
with boxes → lines → paragraphs in reading order, typed as headings, list
items, page furniture — plus tables reconstructed from ruling lines.

Coordinates are display points (top-left origin, rotation applied).
"""

from __future__ import annotations

import ctypes
import statistics
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from textlens.core.result import SOURCE_NATIVE, TABLE, BBox, Block, Page, PageProvenance, Table, TableCell, Word
from textlens.documents.layout import Item, build_blocks, finalize_order
from textlens.documents.pdf.pdfium import PageTransform, page_transform

_BOLD_FLAG = 1 << 18  # PDF font descriptor ForceBold


def _raw():
    import pypdfium2.raw as raw

    return raw


@dataclass
class _Char:
    ch: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    generated: bool


def _chars(page: Any, xf: PageTransform, include_invisible: bool) -> List[_Char]:
    raw = _raw()
    textpage = page.get_textpage()
    out: List[_Char] = []
    try:
        n = textpage.count_chars()
        left, right, bottom, top = (ctypes.c_double() for _ in range(4))
        flags = ctypes.c_int()
        font_cache: dict = {}
        for i in range(n):
            code = raw.FPDFText_GetUnicode(textpage.raw, i)
            generated = raw.FPDFText_IsGenerated(textpage.raw, i) == 1
            ch = chr(code) if 0 < code < 0x110000 else "�"
            if generated or ch in "\r\n\x00" or ch.isspace():
                out.append(_Char(" " if ch not in "\r\n" else "\n", 0, 0, 0, 0, 0, False, True))
                continue
            if not include_invisible:
                tobj = raw.FPDFText_GetTextObject(textpage.raw, i)
                if tobj:
                    mode = raw.FPDFTextObj_GetTextRenderMode(tobj)
                    if mode in (raw.FPDF_TEXTRENDERMODE_INVISIBLE, raw.FPDF_TEXTRENDERMODE_CLIP):
                        continue
            if not raw.FPDFText_GetCharBox(
                textpage.raw, i, ctypes.byref(left), ctypes.byref(right), ctypes.byref(bottom), ctypes.byref(top)
            ):
                continue
            x0, y0, x1, y1 = xf.rect(left.value, bottom.value, right.value, top.value)
            size = raw.FPDFText_GetFontSize(textpage.raw, i) or (y1 - y0)
            weight = raw.FPDFText_GetFontWeight(textpage.raw, i)
            bold = weight >= 600
            if weight <= 0:
                tobj = raw.FPDFText_GetTextObject(textpage.raw, i)
                key = int(ctypes.cast(tobj, ctypes.c_void_p).value or 0) if tobj else 0
                if key not in font_cache:
                    font_cache[key] = False
                    buf = ctypes.create_string_buffer(128)
                    if raw.FPDFText_GetFontInfo(textpage.raw, i, buf, 128, ctypes.byref(flags)):
                        name = buf.value.decode("latin-1", "ignore").lower()
                        font_cache[key] = bool(flags.value & _BOLD_FLAG) or any(
                            s in name for s in ("bold", "black", "heavy", "semibold", "demi")
                        )
                bold = font_cache[key]
            out.append(_Char(ch, x0, y0, x1, y1, float(size), bold, False))
    finally:
        textpage.close()
    return out


def _words(chars: Sequence[_Char]) -> List[Item]:
    words: List[Item] = []
    cur: List[_Char] = []

    def flush() -> None:
        if not cur:
            return
        text = "".join(c.ch for c in cur)
        box = BBox(min(c.x0 for c in cur), min(c.y0 for c in cur), max(c.x1 for c in cur), max(c.y1 for c in cur))
        sizes = [c.size for c in cur]
        bold = sum(1 for c in cur if c.bold) * 2 > len(cur)
        words.append(Item(text=text, bbox=box, font_size=statistics.median(sizes), bold=bold, words=[Word(text, box, None)]))
        cur.clear()

    for c in chars:
        if c.generated:
            flush()
            continue
        if cur:
            prev = cur[-1]
            size = max(prev.size, 1.0)
            same_line = abs(((c.y0 + c.y1) - (prev.y0 + prev.y1)) / 2.0) < 0.5 * size
            gap = c.x0 - prev.x1
            if not same_line or gap > 0.3 * size or gap < -0.6 * size:
                flush()
        cur.append(c)
    flush()
    return words


# ── ruled tables ─────────────────────────────────────────────────────────────


def _rules(page: Any, xf: PageTransform) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Horizontal ``(y, x0, x1)`` and vertical ``(x, y0, y1)`` ruling lines."""
    raw = _raw()
    horizontal, vertical = [], []
    for obj in page.get_objects(max_depth=4):
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        try:
            x0, y0, x1, y1 = xf.rect(*obj.get_bounds())
        except Exception:
            continue
        w, h = x1 - x0, y1 - y0
        if h <= 2.5 and w >= 10:
            horizontal.append(((y0 + y1) / 2.0, x0, x1))
        elif w <= 2.5 and h >= 6:
            vertical.append(((x0 + x1) / 2.0, y0, y1))
        elif w > 10 and h > 6 and raw.FPDFPath_CountSegments(obj.raw) <= 5:
            # A stroked rectangle: contributes all four edges.
            horizontal += [(y0, x0, x1), (y1, x0, x1)]
            vertical += [(x0, y0, y1), (x1, y0, y1)]
    return horizontal, vertical


def _cluster(values: List[float], tol: float) -> List[float]:
    out: List[float] = []
    for v in sorted(values):
        if out and abs(v - out[-1]) <= tol:
            out[-1] = (out[-1] + v) / 2.0
        else:
            out.append(v)
    return out


def detect_ruled_tables(page: Any, xf: PageTransform, words: Sequence[Item], page_number: int) -> Tuple[List[Table], set]:
    """Reconstruct grid tables from ruling lines; return tables + used word ids."""
    horizontal, vertical = _rules(page, xf)
    if len(horizontal) < 2 or len(vertical) < 2:
        return [], set()
    tables: List[Table] = []
    used: set = set()
    # Group lines into connected grids by bounding-box overlap.
    grids: List[dict] = []
    for y, x0, x1 in horizontal:
        grids.append({"h": [(y, x0, x1)], "v": [], "box": [x0, y, x1, y]})
    for x, y0, y1 in vertical:
        for g in grids:
            gx0, gy0, gx1, gy1 = g["box"]
            if gx0 - 3 <= x <= gx1 + 3 and y0 - 3 <= gy1 and y1 + 3 >= gy0:
                g["v"].append((x, y0, y1))
                g["box"] = [min(gx0, x), min(gy0, y0), max(gx1, x), max(gy1, y1)]
    # Merge grids sharing vertical lines.
    merged: List[dict] = []
    for g in grids:
        if not g["v"]:
            continue
        for m in merged:
            mx0, my0, mx1, my1 = m["box"]
            gx0, gy0, gx1, gy1 = g["box"]
            if gx0 <= mx1 + 3 and gx1 >= mx0 - 3 and gy0 <= my1 + 3 and gy1 >= my0 - 3:
                m["h"] += g["h"]
                m["v"] += g["v"]
                m["box"] = [min(mx0, gx0), min(my0, gy0), max(mx1, gx1), max(my1, gy1)]
                break
        else:
            merged.append({"h": list(g["h"]), "v": list(g["v"]), "box": list(g["box"])})
    for g in merged:
        rows = _cluster([h[0] for h in g["h"]], 2.0)
        cols = _cluster([v[0] for v in g["v"]], 2.0)
        if len(rows) < 2 or len(cols) < 2 or (len(rows) - 1) * (len(cols) - 1) < 2:
            continue
        cells: List[TableCell] = []
        texts = [[[] for _ in range(len(cols) - 1)] for _ in range(len(rows) - 1)]
        for wi, w in enumerate(words):
            cx, cy = w.bbox.center
            if not (cols[0] <= cx <= cols[-1] and rows[0] <= cy <= rows[-1]):
                continue
            r = next((k for k in range(len(rows) - 1) if rows[k] <= cy <= rows[k + 1]), None)
            c = next((k for k in range(len(cols) - 1) if cols[k] <= cx <= cols[k + 1]), None)
            if r is None or c is None:
                continue
            texts[r][c].append(w)
            used.add(wi)
        for r in range(len(rows) - 1):
            for c in range(len(cols) - 1):
                members = sorted(texts[r][c], key=lambda w: (round(w.bbox.y0), w.bbox.x0))
                cells.append(
                    TableCell(
                        row=r,
                        col=c,
                        text=" ".join(m.text for m in members),
                        bbox=BBox(cols[c], rows[r], cols[c + 1], rows[r + 1]).rounded(2),
                        is_header=r == 0,
                    )
                )
        tables.append(
            Table(
                cells=cells,
                page=page_number,
                bbox=BBox(cols[0], rows[0], cols[-1], rows[-1]).rounded(2),
                source=SOURCE_NATIVE,
                confidence=1.0,
                id=f"p{page_number}-t{len(tables)}",
            )
        )
    return tables, used


# ── page extraction ──────────────────────────────────────────────────────────


def extract_page(
    page: Any,
    page_number: int,
    include_invisible: bool = False,
    detect_tables: bool = True,
) -> Page:
    """Extract one open ``pypdfium2.PdfPage`` into a TextLens :class:`Page`.

    ``include_invisible=True`` reads an existing (render mode 3) OCR layer,
    which lets fast profiles reuse a scanner's OCR instead of re-running it.
    """
    xf = page_transform(page)
    width, height = xf.width, xf.height  # display size: rotation and crop applied
    words = _words(_chars(page, xf, include_invisible))
    tables: List[Table] = []
    used: set = set()
    if detect_tables and words:
        tables, used = detect_ruled_tables(page, xf, words, page_number)
    blocks = build_blocks(words, page_number, width, height, SOURCE_NATIVE, unit="word", exclude=used)
    for i, table in enumerate(tables):
        blocks.append(
            Block(
                type=TABLE,
                text=table.text,
                bbox=table.bbox,
                confidence=1.0,
                page=page_number,
                source=SOURCE_NATIVE,
                attributes={"table_index": i},
            )
        )
    # Re-sort so tables sit at their vertical position in reading order.
    blocks = _interleave_tables(blocks)
    for b in blocks:
        if b.source == SOURCE_NATIVE and b.confidence is None:
            b.confidence = 1.0
    finalize_order(blocks, page_number)
    return Page(
        number=page_number,
        width=round(width, 2),
        height=round(height, 2),
        unit="pt",
        blocks=blocks,
        tables=tables,
        provenance=PageProvenance(page=page_number, source=SOURCE_NATIVE, backend="pdfium", confidence=1.0),
        classification="native",
        rotation=int(page.get_rotation() or 0),
    )


def _interleave_tables(blocks: List[Block]) -> List[Block]:
    tables = [b for b in blocks if b.type == TABLE]
    if not tables:
        return blocks
    others = [b for b in blocks if b.type != TABLE]
    for t in sorted(tables, key=lambda b: b.bbox.y0 if b.bbox else 0):
        pos = len(others)
        for k, b in enumerate(others):
            if b.bbox and t.bbox and b.bbox.y0 > t.bbox.y0 and b.bbox.x0 < t.bbox.x1 and b.bbox.x1 > t.bbox.x0:
                pos = k
                break
        others.insert(pos, t)
    return others
