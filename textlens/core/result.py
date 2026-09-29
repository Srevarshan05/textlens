"""
textlens.core.result
────────────────────
The unified TextLens result schema.

Every backend — native PDF extraction, PP-OCR, document VLMs, remote
inference servers — produces the same objects, so application code never
depends on which engine ran::

    Result
      ├── pages: [Page]
      │     ├── blocks: [Block]  (type, text, bbox, confidence, source, model…)
      │     │     └── lines: [Line] └── words: [Word]
      │     ├── tables: [Table]  └── cells: [TableCell]
      │     ├── formulas: [Formula]
      │     └── provenance: PageProvenance (native/ocr/vlm, model, dpi, timings…)
      ├── metadata: dict         (source, mime, page count, PDF info…)
      └── provenance: RunProvenance (version, profile, config/document hash…)

Coordinates
    ``bbox`` is ``(x0, y0, x1, y1)`` with a **top-left origin** in the page's
    own unit: PDF points (1/72") for PDF pages, pixels for images.  Each
    :class:`Page` records ``unit``, ``width`` and ``height``, so boxes can be
    mapped back onto the original document for highlighting and citations.

Serialisation
    ``Result.to_dict()`` / ``Result.from_dict()`` round-trip losslessly; the
    JSON schema is versioned by :data:`SCHEMA_VERSION`.
"""

from __future__ import annotations

import dataclasses
import html as _html
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

SCHEMA_VERSION = "2.0"

# Block types.  Plain strings keep the JSON readable and let adapters add
# their own labels without an enum migration.
TEXT = "text"
TITLE = "title"
HEADING = "heading"
LIST_ITEM = "list_item"
TABLE = "table"
FIGURE = "figure"
FORMULA = "formula"
CAPTION = "caption"
PAGE_HEADER = "page_header"
PAGE_FOOTER = "page_footer"
CODE = "code"
FOOTNOTE = "footnote"
FURNITURE_TYPES = frozenset({PAGE_HEADER, PAGE_FOOTER})

# Content sources, recorded on every block and page.
SOURCE_NATIVE = "native"  # PDF text layer / DOCX / PPTX
SOURCE_OCR = "ocr"  # detection + recognition engine (PP-OCR, …)
SOURCE_VLM = "vlm"  # generative document model (GLM-OCR, …)
SOURCE_FUSED = "fused"  # native and OCR content combined
SOURCE_EMPTY = "empty"  # nothing recognisable on the page


class BBox(NamedTuple):
    """Axis-aligned box ``(x0, y0, x1, y1)``, top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def union(self, other: "BBox") -> "BBox":
        return BBox(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))

    def intersection(self, other: "BBox") -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def iou(self, other: "BBox") -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def scale(self, sx: float, sy: Optional[float] = None) -> "BBox":
        sy = sx if sy is None else sy
        return BBox(self.x0 * sx, self.y0 * sy, self.x1 * sx, self.y1 * sy)

    def rounded(self, ndigits: int = 2) -> "BBox":
        return BBox(*(round(v, ndigits) for v in self))

    @classmethod
    def from_points(cls, points: Iterable[Sequence[float]]) -> "BBox":
        pts = list(points)
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def union_all(cls, boxes: Iterable[Optional["BBox"]]) -> Optional["BBox"]:
        out: Optional[BBox] = None
        for b in boxes:
            if b is None:
                continue
            out = b if out is None else out.union(b)
        return out


def _bbox(value: Any) -> Optional[BBox]:
    if value is None:
        return None
    if isinstance(value, BBox):
        return value
    return BBox(*(float(v) for v in value))


def _weighted_confidence(items: Iterable[Tuple[Optional[float], int]]) -> Optional[float]:
    """Mean confidence weighted by text length; ``None`` if nothing is scored."""
    total = 0.0
    weight = 0
    for conf, n in items:
        if conf is None:
            continue
        w = max(1, n)
        total += conf * w
        weight += w
    return round(total / weight, 4) if weight else None


# ── Text hierarchy ───────────────────────────────────────────────────────────


@dataclass
class Word:
    text: str
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None


@dataclass
class Line:
    text: str
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    words: List[Word] = field(default_factory=list)
    polygon: Optional[List[Tuple[float, float]]] = None
    font_size: Optional[float] = None
    bold: Optional[bool] = None

    def split_words(self) -> List[Word]:
        """Words, falling back to whitespace-split tokens without boxes."""
        if self.words:
            return self.words
        return [Word(tok, None, self.confidence) for tok in self.text.split()]


@dataclass
class Block:
    """A layout region: paragraph, heading, list item, figure, formula…"""

    type: str = TEXT
    text: str = ""
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    lines: List[Line] = field(default_factory=list)
    page: int = 1
    source: str = SOURCE_OCR
    model: Optional[str] = None
    backend: Optional[str] = None
    level: Optional[int] = None  # heading level (1 = top)
    order: Optional[int] = None  # reading-order index within the page
    id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> List[Word]:
        out: List[Word] = []
        for line in self.lines:
            out.extend(line.split_words())
        if not self.lines and self.text:
            out.extend(Word(t, None, self.confidence) for t in self.text.split())
        return out


# ── Tables & formulas ────────────────────────────────────────────────────────


@dataclass
class TableCell:
    row: int
    col: int
    text: str = ""
    row_span: int = 1
    col_span: int = 1
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    is_header: bool = False


@dataclass
class Table:
    cells: List[TableCell] = field(default_factory=list)
    page: int = 1
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    source: str = SOURCE_OCR
    model: Optional[str] = None
    caption: Optional[str] = None
    id: Optional[str] = None

    @property
    def n_rows(self) -> int:
        return max((c.row + c.row_span for c in self.cells), default=0)

    @property
    def n_cols(self) -> int:
        return max((c.col + c.col_span for c in self.cells), default=0)

    @classmethod
    def from_grid(cls, rows: Sequence[Sequence[str]], header: bool = True, **kwargs: Any) -> "Table":
        cells = [
            TableCell(row=r, col=c, text=str(val), is_header=header and r == 0)
            for r, row in enumerate(rows)
            for c, val in enumerate(row)
        ]
        return cls(cells=cells, **kwargs)

    def grid(self) -> List[List[str]]:
        """Dense row-major grid; spanned cells repeat their text."""
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            for r in range(c.row, c.row + c.row_span):
                for k in range(c.col, c.col + c.col_span):
                    grid[r][k] = c.text
        return grid

    @property
    def text(self) -> str:
        return "\n".join("\t".join(row) for row in self.grid())

    def to_markdown(self) -> str:
        grid = self.grid()
        if not grid:
            return ""

        def esc(s: str) -> str:
            return s.replace("|", "\\|").replace("\n", " ").strip()

        width = len(grid[0])
        lines = ["| " + " | ".join(esc(v) for v in grid[0]) + " |", "|" + "---|" * width]
        lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in grid[1:]]
        return "\n".join(lines)

    def to_csv(self) -> str:
        import csv
        import io

        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(self.grid())
        return buf.getvalue()

    def to_html(self) -> str:
        rows: Dict[int, List[TableCell]] = {}
        for c in sorted(self.cells, key=lambda c: (c.row, c.col)):
            rows.setdefault(c.row, []).append(c)
        out = ["<table>"]
        for r in sorted(rows):
            out.append("<tr>")
            for c in rows[r]:
                tag = "th" if c.is_header else "td"
                span = ""
                if c.row_span > 1:
                    span += f' rowspan="{c.row_span}"'
                if c.col_span > 1:
                    span += f' colspan="{c.col_span}"'
                out.append(f"<{tag}{span}>{_html.escape(c.text)}</{tag}>")
            out.append("</tr>")
        out.append("</table>")
        return "".join(out)

    def to_records(self) -> List[Dict[str, str]]:
        """Rows as dicts keyed by the header row."""
        grid = self.grid()
        if len(grid) < 2:
            return []
        header = [h or f"col_{i}" for i, h in enumerate(grid[0])]
        return [dict(zip(header, row)) for row in grid[1:]]


@dataclass
class Formula:
    latex: str
    page: int = 1
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    source: str = SOURCE_VLM
    model: Optional[str] = None
    display: bool = True


# ── Provenance ───────────────────────────────────────────────────────────────


@dataclass
class Attempt:
    """One recognition attempt on a page (the fallback chain is a list)."""

    model: str
    backend: Optional[str] = None
    confidence: Optional[float] = None
    accepted: bool = False
    reason: str = ""
    elapsed_ms: Optional[float] = None


@dataclass
class PageProvenance:
    page: int = 1
    source: str = SOURCE_OCR
    model: Optional[str] = None
    model_revision: Optional[str] = None
    backend: Optional[str] = None
    device: Optional[str] = None
    dpi: Optional[float] = None
    confidence: Optional[float] = None
    reasons: List[str] = field(default_factory=list)  # why this page was OCR'd
    routing: Optional[str] = None  # human-readable routing explanation
    attempts: List[Attempt] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    needs_review: bool = False
    cached: bool = False


@dataclass
class RunProvenance:
    """Pipeline fingerprint: enough to reproduce or audit a result."""

    textlens_version: str = ""
    schema_version: str = SCHEMA_VERSION
    profile: Optional[str] = None
    document_hash: Optional[str] = None
    config_hash: Optional[str] = None
    source: Optional[str] = None
    models: List[str] = field(default_factory=list)
    backends: List[str] = field(default_factory=list)
    device: Optional[str] = None
    created_at: Optional[str] = None
    elapsed_ms: Optional[float] = None
    routing: Optional[Dict[str, Any]] = None


# ── Pages & results ──────────────────────────────────────────────────────────


@dataclass
class Page:
    number: int
    width: float = 0.0
    height: float = 0.0
    unit: str = "px"  # "pt" for PDF pages, "px" for images
    blocks: List[Block] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    formulas: List[Formula] = field(default_factory=list)
    provenance: PageProvenance = field(default_factory=PageProvenance)
    classification: Optional[str] = None  # native | scanned | mixed | image | …
    rotation: int = 0
    markdown: Optional[str] = None  # engine-native Markdown (VLMs), if any

    def ordered_blocks(self, include_furniture: bool = True) -> List[Block]:
        blocks = [b for b in self.blocks if include_furniture or b.type not in FURNITURE_TYPES]
        return sorted(blocks, key=lambda b: (b.order if b.order is not None else 1_000_000))

    @property
    def text(self) -> str:
        parts: List[str] = []
        for b in self.ordered_blocks():
            if b.type == TABLE and "table_index" in b.attributes:
                idx = b.attributes["table_index"]
                if 0 <= idx < len(self.tables):
                    parts.append(self.tables[idx].text)
                    continue
            if b.text:
                parts.append(b.text)
        return "\n".join(parts)

    @property
    def lines(self) -> List[Line]:
        out: List[Line] = []
        for b in self.ordered_blocks():
            out.extend(b.lines or ([Line(b.text, b.bbox, b.confidence)] if b.text else []))
        return out

    @property
    def words(self) -> List[Word]:
        out: List[Word] = []
        for b in self.ordered_blocks():
            out.extend(b.words)
        return out

    @property
    def confidence(self) -> Optional[float]:
        scored = _weighted_confidence((b.confidence, len(b.text)) for b in self.blocks)
        return scored if scored is not None else self.provenance.confidence

    def to_markdown(self, include_furniture: bool = False) -> str:
        from textlens.core.export import page_to_markdown

        return page_to_markdown(self, include_furniture=include_furniture)


@dataclass
class Result:
    """The unified output of every TextLens OCR / document call."""

    pages: List[Page] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: RunProvenance = field(default_factory=RunProvenance)
    extraction: Optional[Dict[str, Any]] = None  # structured extraction output

    # ── convenient views ────────────────────────────────────────────────
    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def blocks(self) -> List[Block]:
        return [b for p in self.pages for b in p.ordered_blocks()]

    @property
    def lines(self) -> List[Line]:
        return [ln for p in self.pages for ln in p.lines]

    @property
    def words(self) -> List[Word]:
        return [w for p in self.pages for w in p.words]

    @property
    def tables(self) -> List[Table]:
        return [t for p in self.pages for t in p.tables]

    @property
    def formulas(self) -> List[Formula]:
        return [f for p in self.pages for f in p.formulas]

    @property
    def layout(self) -> List[Dict[str, Any]]:
        """Flat layout map: one entry per block with page, type, bbox, order."""
        return [
            {
                "page": b.page,
                "type": b.type,
                "bbox": list(b.bbox) if b.bbox else None,
                "order": b.order,
                "id": b.id,
                "source": b.source,
            }
            for b in self.blocks
        ]

    @property
    def confidence(self) -> Optional[float]:
        return _weighted_confidence((p.confidence, len(p.text)) for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def pages_needing_review(self) -> List[int]:
        return [p.number for p in self.pages if p.provenance.needs_review]

    # ── dunder helpers ──────────────────────────────────────────────────
    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.pages)

    def __iter__(self) -> Iterator[Page]:
        return iter(self.pages)

    def __getitem__(self, index: int) -> Page:
        return self.pages[index]

    def __repr__(self) -> str:
        conf = self.confidence
        conf_s = f"{conf:.3f}" if conf is not None else "n/a"
        src = self.metadata.get("source_name") or self.provenance.source
        return f"<Result source={src!r} pages={len(self.pages)} chars={len(self.text)} confidence={conf_s}>"

    # ── exporters ───────────────────────────────────────────────────────
    def to_text(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    def to_json(self, indent: Optional[int] = 2, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, **kwargs)

    def to_markdown(self, page_markers: bool = False, include_furniture: bool = False) -> str:
        from textlens.core.export import result_to_markdown

        return result_to_markdown(self, page_markers=page_markers, include_furniture=include_furniture)

    def to_html(self, full_document: bool = True) -> str:
        from textlens.core.export import result_to_html

        return result_to_html(self, full_document=full_document)

    def to_csv(self) -> str:
        """All tables as CSV blocks separated by blank lines."""
        return "\n".join(t.to_csv() for t in self.tables)

    def to_chunks(self, **kwargs: Any) -> List["Chunk"]:
        from textlens.core.export import chunk_result

        return chunk_result(self, **kwargs)

    def save(self, directory: Any, formats: Sequence[str] = ("json", "md", "txt")) -> Dict[str, str]:
        """Write result artifacts (``document.json``, ``document.md``…)."""
        from textlens.core.export import save_artifacts

        return save_artifacts(self, directory, formats)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Result":
        return _result_from_dict(data)

    @classmethod
    def from_json(cls, text: str) -> "Result":
        return cls.from_dict(json.loads(text))


@dataclass
class Chunk:
    """A retrieval-ready slice of a result with citation provenance."""

    text: str
    index: int
    pages: List[int] = field(default_factory=list)
    bboxes: List[Dict[str, Any]] = field(default_factory=list)  # [{page, bbox}]
    section: List[str] = field(default_factory=list)  # heading path
    block_ids: List[str] = field(default_factory=list)
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


# ── (de)serialisation helpers ────────────────────────────────────────────────


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, BBox):
        return [round(float(v), 3) for v in obj]
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def _build(cls: Any, data: Dict[str, Any], nested: Dict[str, Any]) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in names:
            continue
        if key in nested and value is not None:
            conv = nested[key]
            kwargs[key] = [conv(v) for v in value] if isinstance(value, list) else conv(value)
        elif key in ("bbox",):
            kwargs[key] = _bbox(value)
        elif key == "polygon" and value is not None:
            kwargs[key] = [tuple(p) for p in value]
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _word(d: Dict[str, Any]) -> Word:
    return _build(Word, d, {})


def _line(d: Dict[str, Any]) -> Line:
    return _build(Line, d, {"words": _word})


def _block(d: Dict[str, Any]) -> Block:
    return _build(Block, d, {"lines": _line})


def _cell(d: Dict[str, Any]) -> TableCell:
    return _build(TableCell, d, {})


def _table(d: Dict[str, Any]) -> Table:
    return _build(Table, d, {"cells": _cell})


def _formula(d: Dict[str, Any]) -> Formula:
    return _build(Formula, d, {})


def _attempt(d: Dict[str, Any]) -> Attempt:
    return _build(Attempt, d, {})


def _page_prov(d: Dict[str, Any]) -> PageProvenance:
    return _build(PageProvenance, d, {"attempts": _attempt})


def _page(d: Dict[str, Any]) -> Page:
    return _build(
        Page,
        d,
        {"blocks": _block, "tables": _table, "formulas": _formula, "provenance": _page_prov},
    )


def _result_from_dict(d: Dict[str, Any]) -> Result:
    return _build(
        Result,
        d,
        {"pages": _page, "provenance": lambda v: _build(RunProvenance, v, {})},
    )
