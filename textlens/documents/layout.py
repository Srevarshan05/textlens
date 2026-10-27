"""
textlens.documents.layout
─────────────────────────
Engine-agnostic layout analysis: reading order, lines, paragraphs, headings.

The same code orders native PDF words and OCR line detections, so a scanned
page and its born-digital twin produce comparable blocks.

Reading order — recursive XY-cut
    The region is split at the widest whitespace gap in either projection:
    horizontal gaps give top→bottom order, vertical gaps give columns read
    left→right.  A vertical cut is only accepted when *both* sides look like
    prose columns (several lines, several words per line).  This stops
    tabular layouts — invoices, forms, key/value sheets — from being read
    column-by-column, which would separate labels from their values; they
    fall through to row-major (Y-interleaved) order instead.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from textlens.core.result import (
    HEADING,
    LIST_ITEM,
    PAGE_FOOTER,
    PAGE_HEADER,
    TEXT,
    TITLE,
    BBox,
    Block,
    Line,
    Word,
)

_LIST_RE = re.compile(r"^\s*([-*+•▪◦●‣]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|[ivxIVX]{1,4}[.)])\s+")
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)


@dataclass
class Item:
    """A positioned text unit (word for native PDFs, line for OCR)."""

    text: str
    bbox: BBox
    confidence: Optional[float] = None
    font_size: Optional[float] = None
    bold: bool = False
    words: List[Word] = field(default_factory=list)
    polygon: Optional[List[Tuple[float, float]]] = None

    @property
    def height(self) -> float:
        return self.bbox.height


# ── XY-cut ───────────────────────────────────────────────────────────────────


def _gaps(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Whitespace gaps ``(start, end)`` between merged 1-D intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]


def _row_count(items: Sequence[Item]) -> int:
    """Number of distinct text rows (vertical clusters)."""
    return len(_gaps([(i.bbox.y0, i.bbox.y1) for i in items])) + (1 if items else 0)


def _prose_like(items: Sequence[Item], unit: str) -> bool:
    rows = _row_count(items)
    if rows < 3:
        return False
    tokens = sum(len(i.text.split()) for i in items) if unit == "line" else len(items)
    return tokens / rows >= 3.0


def xy_cut(items: Sequence[Item], unit: str = "word", depth: int = 0) -> List[List[int]]:
    """Return item index groups (regions) in reading order."""
    index = list(range(len(items)))
    out: List[List[int]] = []
    _xy(items, index, unit, out, depth)
    return out


def _xy(items: Sequence[Item], idx: List[int], unit: str, out: List[List[int]], depth: int) -> None:
    if len(idx) <= 1 or depth > 40:
        if idx:
            out.append(idx)
        return
    sub = [items[i] for i in idx]
    heights = [it.height for it in sub if it.height > 0]
    med_h = statistics.median(heights) if heights else 10.0
    min_gap_y = max(1.0, 0.9 * med_h)
    min_gap_x = max(1.0, 1.2 * med_h)

    ygaps = [(a, b) for a, b in _gaps([(it.bbox.y0, it.bbox.y1) for it in sub]) if b - a >= min_gap_y]
    xgaps = [(a, b) for a, b in _gaps([(it.bbox.x0, it.bbox.x1) for it in sub]) if b - a >= min_gap_x]

    # Try the widest acceptable vertical (column) cut first when it is
    # clearly a gutter between two prose columns.
    for a, b in sorted(xgaps, key=lambda g: g[1] - g[0], reverse=True):
        mid = (a + b) / 2.0
        left = [i for i in idx if items[i].bbox.x1 <= mid]
        right = [i for i in idx if items[i].bbox.x0 >= mid]
        if len(left) + len(right) != len(idx):
            continue
        region_w = max(it.bbox.x1 for it in sub) - min(it.bbox.x0 for it in sub)
        lw = max(items[i].bbox.x1 for i in left) - min(items[i].bbox.x0 for i in left)
        rw = max(items[i].bbox.x1 for i in right) - min(items[i].bbox.x0 for i in right)
        if min(lw, rw) < 0.2 * region_w:
            continue
        if not (_prose_like([items[i] for i in left], unit) and _prose_like([items[i] for i in right], unit)):
            continue
        # Prefer a horizontal cut if one is wider (e.g. title above columns).
        widest_y = max((g[1] - g[0] for g in ygaps), default=0.0)
        if widest_y > (b - a) and widest_y >= 1.5 * med_h:
            break
        _xy(items, left, unit, out, depth + 1)
        _xy(items, right, unit, out, depth + 1)
        return

    if ygaps:
        a, b = max(ygaps, key=lambda g: g[1] - g[0])
        mid = (a + b) / 2.0
        top = [i for i in idx if items[i].bbox.y1 <= mid]
        bottom = [i for i in idx if items[i].bbox.y0 >= mid]
        if top and bottom and len(top) + len(bottom) == len(idx):
            _xy(items, top, unit, out, depth + 1)
            _xy(items, bottom, unit, out, depth + 1)
            return
    out.append(idx)


# ── lines & paragraphs ───────────────────────────────────────────────────────


def group_lines(items: Sequence[Item], idx: Sequence[int]) -> List[List[int]]:
    """Group items into text lines (vertical overlap), each sorted by x."""
    order = sorted(idx, key=lambda i: (items[i].bbox.y0 + items[i].bbox.y1) / 2.0)
    lines: List[List[int]] = []
    spans: List[Tuple[float, float]] = []
    for i in order:
        b = items[i].bbox
        placed = False
        for k in range(len(lines) - 1, max(-1, len(lines) - 4), -1):
            y0, y1 = spans[k]
            overlap = min(y1, b.y1) - max(y0, b.y0)
            if overlap > 0.5 * min(y1 - y0, b.height or 1.0):
                lines[k].append(i)
                spans[k] = (min(y0, b.y0), max(y1, b.y1))
                placed = True
                break
        if not placed:
            lines.append([i])
            spans.append((b.y0, b.y1))
    for line in lines:
        line.sort(key=lambda i: items[i].bbox.x0)
    lines.sort(key=lambda ln: min(items[i].bbox.y0 for i in ln))
    return lines


def _join_words(items: Sequence[Item], line: Sequence[int]) -> str:
    parts: List[str] = []
    prev: Optional[Item] = None
    for i in line:
        it = items[i]
        if prev is not None:
            gap = it.bbox.x0 - prev.bbox.x1
            size = prev.font_size or prev.height or 10.0
            # Wide gaps inside a line (tab stops, table columns) become tabs.
            parts.append("\t" if gap > 2.5 * size else " ")
        parts.append(it.text)
        prev = it
    return "".join(parts).strip()


def _make_line(items: Sequence[Item], line: Sequence[int]) -> Line:
    members = [items[i] for i in line]
    words: List[Word] = []
    for m in members:
        words.extend(m.words or [Word(m.text, m.bbox, m.confidence)])
    confs = [(m.confidence, len(m.text)) for m in members if m.confidence is not None]
    conf = (sum(c * n for c, n in confs) / max(1, sum(n for _, n in confs))) if confs else None
    sizes = [m.font_size for m in members if m.font_size]
    return Line(
        text=_join_words(items, line),
        bbox=BBox.union_all(m.bbox for m in members),
        confidence=round(conf, 4) if conf is not None else None,
        words=words,
        font_size=statistics.median(sizes) if sizes else None,
        bold=sum(1 for m in members if m.bold) * 2 > len(members),
        polygon=members[0].polygon if len(members) == 1 else None,
    )


def split_paragraphs(lines: List[Line]) -> List[List[Line]]:
    """Split a region's lines into paragraphs at larger gaps or style changes."""
    if not lines:
        return []
    heights = [ln.bbox.height for ln in lines if ln.bbox and ln.bbox.height > 0]
    med_h = statistics.median(heights) if heights else 10.0
    gaps = [
        lines[i + 1].bbox.y0 - lines[i].bbox.y1
        for i in range(len(lines) - 1)
        if lines[i].bbox and lines[i + 1].bbox
    ]
    positive = [g for g in gaps if g > 0]
    base_gap = statistics.median(positive) if positive else 0.3 * med_h
    paras: List[List[Line]] = [[lines[0]]]
    for prev, cur in zip(lines, lines[1:]):
        gap = (cur.bbox.y0 - prev.bbox.y1) if (cur.bbox and prev.bbox) else 0.0
        size_change = (
            prev.font_size and cur.font_size and abs(prev.font_size - cur.font_size) > 0.15 * max(prev.font_size, cur.font_size)
        )
        style_change = bool(prev.bold) != bool(cur.bold)
        list_start = bool(_LIST_RE.match(cur.text))
        if gap > max(base_gap * 1.8, 0.6 * med_h) or size_change or style_change or list_start:
            paras.append([cur])
        else:
            paras[-1].append(cur)
    return paras


# ── block assembly ───────────────────────────────────────────────────────────


def build_blocks(
    items: Sequence[Item],
    page_number: int,
    page_width: float,
    page_height: float,
    source: str,
    unit: str = "word",
    model: Optional[str] = None,
    backend: Optional[str] = None,
    exclude: Optional[set] = None,
) -> List[Block]:
    """Turn positioned items into ordered, typed blocks."""
    exclude = exclude or set()
    keep = [i for i in range(len(items)) if i not in exclude and items[i].text.strip()]
    if not keep:
        return []
    sub = [items[i] for i in keep]
    regions = xy_cut(sub, unit=unit)

    raw_blocks: List[Tuple[List[Line], BBox]] = []
    for region in regions:
        lines = [_make_line(sub, ln) for ln in group_lines(sub, region)]
        for para in split_paragraphs(lines):
            box = BBox.union_all(ln.bbox for ln in para)
            if box is not None:
                raw_blocks.append((para, box))

    body_size = _body_size([ln for para, _ in raw_blocks for ln in para])
    blocks: List[Block] = []
    for para, box in raw_blocks:
        text = "\n".join(ln.text for ln in para)
        btype, level = _classify_block(para, text, box, body_size, page_width, page_height, page_number, is_first=not blocks)
        confs = [(ln.confidence, len(ln.text)) for ln in para if ln.confidence is not None]
        conf = (sum(c * n for c, n in confs) / max(1, sum(n for _, n in confs))) if confs else None
        blocks.append(
            Block(
                type=btype,
                text=text,
                bbox=box.rounded(2),
                confidence=round(conf, 4) if conf is not None else None,
                lines=para,
                page=page_number,
                source=source,
                model=model,
                backend=backend,
                level=level,
            )
        )
    return blocks


def _body_size(lines: List[Line]) -> float:
    weighted: Dict[float, int] = {}
    for ln in lines:
        size = round(ln.font_size or (ln.bbox.height if ln.bbox else 0.0), 1)
        if size > 0:
            weighted[size] = weighted.get(size, 0) + len(ln.text)
    return max(weighted, key=weighted.get) if weighted else 10.0  # type: ignore[arg-type]


def _classify_block(
    para: List[Line],
    text: str,
    box: BBox,
    body: float,
    page_w: float,
    page_h: float,
    page_number: int,
    is_first: bool,
) -> Tuple[str, Optional[int]]:
    sizes = [ln.font_size or (ln.bbox.height if ln.bbox else 0.0) for ln in para]
    size = statistics.median(sizes) if sizes else body
    short = len(text) <= 160 and len(para) <= 3
    margin = 0.07 * page_h
    if page_h and (box.y1 <= margin or box.y0 >= page_h - margin) and len(text) <= 80 and len(para) == 1:
        if _PAGE_NUMBER_RE.match(text):
            return (PAGE_HEADER if box.y1 <= margin else PAGE_FOOTER), None
    if _LIST_RE.match(text):
        return LIST_ITEM, None
    ratio = size / body if body else 1.0
    if short and not text.rstrip().endswith((".", ",", ";")):
        if ratio >= 1.6 and page_number == 1 and box.y0 < page_h * 0.35:
            return TITLE, 1
        if ratio >= 1.15:
            return HEADING, (2 if ratio >= 1.45 else 3)
        if all(ln.bold for ln in para) and len(text) <= 100:
            return HEADING, 4
    return TEXT, None


def finalize_order(blocks: List[Block], page_number: int) -> List[Block]:
    """Assign reading-order indices and stable ids in place."""
    for i, b in enumerate(blocks):
        b.order = i
        b.id = f"p{page_number}-b{i}"
    return blocks


def mark_repeated_furniture(pages_blocks: List[List[Block]], page_heights: List[float], min_pages: int = 3) -> None:
    """Label running headers/footers repeated across many pages."""
    if len(pages_blocks) < min_pages:
        return
    counts: Dict[Tuple[str, str], int] = {}
    for blocks, h in zip(pages_blocks, page_heights):
        seen = set()
        for b in blocks:
            if not b.bbox or not h:
                continue
            zone = "top" if b.bbox.y1 <= 0.1 * h else "bottom" if b.bbox.y0 >= 0.9 * h else None
            if zone and len(b.text) <= 120:
                key = (zone, re.sub(r"\d+", "#", b.text.strip().lower()))
                if key not in seen:
                    counts[key] = counts.get(key, 0) + 1
                    seen.add(key)
    threshold = max(min_pages, int(0.5 * len(pages_blocks)))
    repeated = {k for k, v in counts.items() if v >= threshold}
    for blocks, h in zip(pages_blocks, page_heights):
        for b in blocks:
            if not b.bbox or not h:
                continue
            zone = "top" if b.bbox.y1 <= 0.1 * h else "bottom" if b.bbox.y0 >= 0.9 * h else None
            if zone and (zone, re.sub(r"\d+", "#", b.text.strip().lower())) in repeated:
                b.type = PAGE_HEADER if zone == "top" else PAGE_FOOTER
