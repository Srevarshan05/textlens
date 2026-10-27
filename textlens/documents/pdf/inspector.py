"""
textlens.documents.pdf.inspector
────────────────────────────────
Fast PDF classification and per-page OCR routing.

The inspector answers, for every page and in a few milliseconds:

* Is there a usable native text layer?          → extract it, no OCR
* Is the page a scan (image covering the page)? → OCR it
* Is there an invisible OCR layer from a scanner? → reuse or re-OCR
* Is the text layer garbled (broken encoding)?  → OCR it
* Is the "text" really vector outlines?         → OCR it
* Is the page blank?                            → skip it
* Does it look complex (columns, tables, formulas)? → routing hint

It reads PDF *objects* (text, image, path, form) and the PDFium text page;
it never rasterises.  Per-page reason codes follow Firecrawl pdf-inspector's
vocabulary (``scanned``, ``no_text``, ``vector_text``,
``invisible_text_layer``, ``suspected_garbled_text``) so results are easy to
compare; the implementation is TextLens' own, built on pypdfium2.
"""

from __future__ import annotations

import ctypes
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from textlens.core.quality import assess_text
from textlens.documents.pdf.pdfium import PDFIUM_LOCK, PageTransform, open_pdf, page_transform

logger = logging.getLogger("textlens.pdf.inspector")

# Reason codes (why a page needs OCR)
REASON_SCANNED = "scanned"
REASON_NO_TEXT = "no_text"
REASON_VECTOR_TEXT = "vector_text"
REASON_INVISIBLE_LAYER = "invisible_text_layer"
REASON_GARBLED = "suspected_garbled_text"
REASON_SPARSE_OVER_IMAGE = "sparse_text_over_image"

# Page kinds
KIND_NATIVE = "native"
KIND_MIXED = "mixed"
KIND_SCANNED = "scanned"
KIND_BROKEN = "broken_encoding"
KIND_VECTOR = "vector_text"
KIND_EMPTY = "empty"
KIND_GRAPHICS = "graphics"

# Document types
DOC_TEXT_BASED = "text_based"
DOC_SCANNED = "scanned"
DOC_IMAGE_BASED = "image_based"
DOC_MIXED = "mixed"
DOC_EMPTY = "empty"

_MATH_FONT = re.compile(r"(CMMI|CMSY|CMEX|MSAM|MSBM|Symbol|STIX.*Math|CambriaMath|Cambria-Math|MTSY|MTEX|Euclid|LatinModernMath|XITSMath|Asana|RMTMI|TeX-)", re.I)
_MATH_CHARS = set("∑∏∫∮√∞≤≥±×÷∂∇∈∉⊂⊆⊃∪∩→←⇒⇔≈≠≡∝∀∃∧∨¬αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")
_GRID = 48  # coverage grid resolution


@dataclass
class InspectionConfig:
    """Thresholds for page classification (all tunable)."""

    min_text_chars: int = 20  # visible alphanumerics for a page to count as text
    scan_image_coverage: float = 0.5  # image covering this much of the page ⇒ scan-like
    mixed_image_coverage: float = 0.2  # native page with this much image ⇒ "mixed"
    sparse_text_chars: int = 200  # text below this over a full-page image ⇒ scan + stamp
    vector_path_segments: int = 1500  # path segments with no text ⇒ outlined glyphs
    unicode_error_ratio: float = 0.1  # share of chars PDFium could not map
    max_chars_for_quality: int = 20000  # cap text scanned by quality checks


@dataclass
class PageInspection:
    """Everything the inspector learned about one page (1-indexed)."""

    number: int
    width: float
    height: float
    rotation: int = 0
    kind: str = KIND_NATIVE
    needs_ocr: bool = False
    reasons: List[str] = field(default_factory=list)
    text_chars: int = 0  # non-whitespace chars in the text layer
    visible_alnum: int = 0  # alphanumerics drawn visibly
    invisible_chars: int = 0  # chars drawn with render mode 3/7 (OCR layers)
    unicode_errors: int = 0
    text_objects: int = 0
    image_objects: int = 0
    path_objects: int = 0
    form_objects: int = 0
    path_segments: int = 0
    image_coverage: float = 0.0
    text_coverage: float = 0.0
    largest_image_px: Optional[List[int]] = None
    image_boxes: List[List[float]] = field(default_factory=list)  # top-left pt boxes of images
    image_dpi: Optional[float] = None  # effective DPI of the largest image
    horizontal_rules: int = 0
    vertical_rules: int = 0
    math_signals: int = 0
    quality_signals: List[str] = field(default_factory=list)
    has_ocr_layer: bool = False
    elapsed_ms: float = 0.0

    @property
    def likely_table(self) -> bool:
        return self.horizontal_rules >= 3 and self.vertical_rules >= 2

    @property
    def likely_formulas(self) -> bool:
        return self.math_signals >= 3

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["likely_table"] = self.likely_table
        data["likely_formulas"] = self.likely_formulas
        return data


@dataclass
class DocumentInspection:
    """Document-level classification plus per-page detail."""

    page_count: int
    pdf_type: str
    confidence: float
    pages: List[PageInspection]
    pages_needing_ocr: List[int]
    reasons_by_page: Dict[int, List[str]]
    sampled: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def ocr_ratio(self) -> float:
        return len(self.pages_needing_ocr) / self.page_count if self.page_count else 0.0

    def summary(self) -> str:
        n = len(self.pages_needing_ocr)
        scope = f"{len(self.pages)} of {self.page_count} pages sampled" if self.sampled else f"{self.page_count} pages"
        return (
            f"{self.pdf_type} PDF ({scope}); {n} page(s) need OCR; "
            f"confidence {self.confidence:.2f}; inspected in {self.elapsed_ms:.1f} ms"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_count": self.page_count,
            "pdf_type": self.pdf_type,
            "confidence": round(self.confidence, 3),
            "pages_needing_ocr": self.pages_needing_ocr,
            "reasons_by_page": {str(k): v for k, v in self.reasons_by_page.items()},
            "sampled": self.sampled,
            "metadata": self.metadata,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "pages": [p.to_dict() for p in self.pages],
        }


def _raw():
    import pypdfium2.raw as raw

    return raw


def _mark(grid: List[bool], box: Sequence[float], width: float, height: float) -> None:
    x0, y0, x1, y1 = box
    if width <= 0 or height <= 0:
        return
    c0 = max(0, min(_GRID - 1, int(x0 / width * _GRID)))
    c1 = max(0, min(_GRID - 1, int(max(x0, x1 - 1e-6) / width * _GRID)))
    r0 = max(0, min(_GRID - 1, int(y0 / height * _GRID)))
    r1 = max(0, min(_GRID - 1, int(max(y0, y1 - 1e-6) / height * _GRID)))
    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
        return
    for r in range(r0, r1 + 1):
        base = r * _GRID
        for c in range(c0, c1 + 1):
            grid[base + c] = True


def analyze_page(page: Any, number: int, config: Optional[InspectionConfig] = None) -> PageInspection:
    """Inspect one open ``pypdfium2.PdfPage`` (caller holds ``PDFIUM_LOCK``)."""
    cfg = config or InspectionConfig()
    raw = _raw()
    t0 = time.perf_counter()
    xf: PageTransform = page_transform(page)
    # get_size() is already the display size (/Rotate applied, CropBox).
    disp_w, disp_h = xf.width, xf.height
    info = PageInspection(number=number, width=round(disp_w, 2), height=round(disp_h, 2), rotation=int(page.get_rotation() or 0))

    image_grid = [False] * (_GRID * _GRID)
    invisible_objects = 0
    form_boxes: List[Any] = []
    largest_px = 0
    math_fonts_seen = 0
    fonts_checked = 0

    for obj in page.get_objects(max_depth=4):
        kind = obj.type
        if kind == raw.FPDF_PAGEOBJ_TEXT:
            info.text_objects += 1
            mode = raw.FPDFTextObj_GetTextRenderMode(obj.raw)
            if mode in (raw.FPDF_TEXTRENDERMODE_INVISIBLE, raw.FPDF_TEXTRENDERMODE_CLIP):
                invisible_objects += 1
            if fonts_checked < 64:
                fonts_checked += 1
                font = raw.FPDFTextObj_GetFont(obj.raw)
                if font:
                    buf = ctypes.create_string_buffer(128)
                    n = raw.FPDFFont_GetBaseFontName(font, buf, 128)
                    if n and _MATH_FONT.search(buf.value.decode("latin-1", "ignore")):
                        math_fonts_seen += 1
        elif kind == raw.FPDF_PAGEOBJ_IMAGE:
            info.image_objects += 1
            try:
                l, b, r, t = obj.get_bounds()
            except Exception:
                continue
            box = xf.rect(l, b, r, t)
            if obj.level > 0 and form_boxes:
                # Children of form XObjects may be reported in form space;
                # clip to the enclosing top-level form's page box.
                fx0, fy0, fx1, fy1 = form_boxes[-1]
                box = (max(box[0], fx0), max(box[1], fy0), min(box[2], fx1), min(box[3], fy1))
            _mark(image_grid, box, disp_w, disp_h)
            if (box[2] - box[0]) > 1 and (box[3] - box[1]) > 1 and len(info.image_boxes) < 64:
                info.image_boxes.append([round(max(0.0, v), 2) for v in box])
            try:
                pw, ph = obj.get_px_size()
            except Exception:
                pw = ph = 0
            if pw * ph > largest_px:
                largest_px = pw * ph
                info.largest_image_px = [int(pw), int(ph)]
                bw = max(box[2] - box[0], 1e-3)
                info.image_dpi = round(pw / (bw / 72.0), 1)
        elif kind == raw.FPDF_PAGEOBJ_PATH:
            info.path_objects += 1
            info.path_segments += max(0, raw.FPDFPath_CountSegments(obj.raw))
            try:
                l, b, r, t = obj.get_bounds()
                x0, y0, x1, y1 = xf.rect(l, b, r, t)
                w, h = x1 - x0, y1 - y0
                if h <= 2.0 and w >= 20:
                    info.horizontal_rules += 1
                elif w <= 2.0 and h >= 8:
                    info.vertical_rules += 1
            except Exception:
                pass
        elif kind == raw.FPDF_PAGEOBJ_FORM:
            info.form_objects += 1
            if obj.level == 0:
                try:
                    form_boxes.append(xf.rect(*obj.get_bounds()))
                except Exception:
                    pass

    info.image_coverage = round(sum(image_grid) / float(len(image_grid)), 3)

    # ── text layer ──────────────────────────────────────────────────────
    textpage = page.get_textpage()
    try:
        n_chars = textpage.count_chars()
        text = textpage.get_text_range() if n_chars else ""
        text_grid = [False] * (_GRID * _GRID)
        visible_alnum = 0
        invisible_chars = 0
        unicode_errors = 0
        math_chars = 0
        left, right, bottom, top = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        # Sample large pages to keep inspection O(ms): every k-th char.
        stride = max(1, n_chars // 4000)
        for i in range(0, n_chars, stride):
            if raw.FPDFText_IsGenerated(textpage.raw, i) == 1:
                continue
            code = raw.FPDFText_GetUnicode(textpage.raw, i)
            ch = chr(code) if 0 < code < 0x110000 else "�"
            if ch.isspace():
                continue
            if raw.FPDFText_HasUnicodeMapError(textpage.raw, i) == 1:
                unicode_errors += stride
            if ch in _MATH_CHARS:
                math_chars += stride
            invisible = False
            if invisible_objects:
                tobj = raw.FPDFText_GetTextObject(textpage.raw, i)
                if tobj:
                    mode = raw.FPDFTextObj_GetTextRenderMode(tobj)
                    invisible = mode in (raw.FPDF_TEXTRENDERMODE_INVISIBLE, raw.FPDF_TEXTRENDERMODE_CLIP)
            if invisible:
                invisible_chars += stride
                continue
            if ch.isalnum():
                visible_alnum += stride
            if raw.FPDFText_GetCharBox(textpage.raw, i, ctypes.byref(left), ctypes.byref(right), ctypes.byref(bottom), ctypes.byref(top)):
                _mark(text_grid, xf.rect(left.value, bottom.value, right.value, top.value), disp_w, disp_h)
    finally:
        textpage.close()

    info.text_chars = sum(1 for c in text if not c.isspace())
    info.visible_alnum = visible_alnum
    info.invisible_chars = invisible_chars
    info.unicode_errors = unicode_errors
    info.text_coverage = round(sum(text_grid) / float(len(text_grid)), 4) if n_chars else 0.0
    info.math_signals = math_chars + 3 * math_fonts_seen

    quality = assess_text(text[: cfg.max_chars_for_quality]) if info.text_chars else None
    if quality is not None:
        info.quality_signals = list(quality.signals)

    _classify(info, cfg, quality_garbled=bool(quality and quality.garbled))
    info.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return info


def _classify(info: PageInspection, cfg: InspectionConfig, quality_garbled: bool) -> None:
    reasons: List[str] = []
    has_visible_text = info.visible_alnum >= cfg.min_text_chars
    covered = info.image_coverage >= cfg.scan_image_coverage

    if info.invisible_chars > 0 and not has_visible_text and (covered or info.image_objects):
        reasons.append(REASON_INVISIBLE_LAYER)
        info.has_ocr_layer = True

    ratio = info.unicode_errors / max(1, info.text_chars)
    if has_visible_text and (quality_garbled or (info.unicode_errors >= 10 and ratio >= cfg.unicode_error_ratio)):
        reasons.append(REASON_GARBLED)

    if not has_visible_text and REASON_INVISIBLE_LAYER not in reasons:
        if info.image_objects and info.image_coverage >= 0.05:
            reasons.append(REASON_SCANNED)
        elif info.path_segments >= cfg.vector_path_segments:
            reasons.append(REASON_VECTOR_TEXT)
        elif info.text_chars > 0:
            reasons.append(REASON_NO_TEXT)  # a few glyphs, nothing trustworthy
    elif has_visible_text and covered and info.visible_alnum < cfg.sparse_text_chars:
        reasons.append(REASON_SPARSE_OVER_IMAGE)

    info.reasons = reasons
    info.needs_ocr = bool(reasons)
    if REASON_GARBLED in reasons:
        info.kind = KIND_BROKEN
    elif REASON_VECTOR_TEXT in reasons:
        info.kind = KIND_VECTOR
    elif reasons:
        info.kind = KIND_SCANNED
    elif not has_visible_text:
        # No text and nothing to OCR: blank, or pure vector graphics.
        info.kind = KIND_GRAPHICS if (info.path_objects or info.image_objects) else KIND_EMPTY
    elif info.image_coverage >= cfg.mixed_image_coverage:
        info.kind = KIND_MIXED
    else:
        info.kind = KIND_NATIVE


def sample_pages(page_count: int, n: int) -> List[int]:
    """First, last and evenly spaced pages (1-indexed), like pdf-inspector."""
    if n <= 0:
        return []
    if n >= page_count:
        return list(range(1, page_count + 1))
    picks = {1, page_count}
    remaining = n - 2
    if remaining > 0 and page_count > 2:
        step = (page_count - 2) / (remaining + 1)
        for i in range(1, remaining + 1):
            picks.add(1 + int(round(step * i)))
    return sorted(picks)[:n] if len(picks) > n else sorted(picks)


def classify_document(pages: Sequence[PageInspection], page_count: int, sampled: bool) -> tuple:
    """Return ``(pdf_type, confidence)`` from inspected pages."""
    if not pages:
        return DOC_EMPTY, 1.0
    ocr_pages = [p for p in pages if p.needs_ocr]
    content_pages = [p for p in pages if p.kind not in (KIND_EMPTY,)]
    if not content_pages:
        return DOC_EMPTY, 1.0
    if not ocr_pages:
        return DOC_TEXT_BASED, round(len(content_pages) / len(pages), 3) if len(pages) else 1.0
    if len(ocr_pages) == len(content_pages):
        scan_like = all(
            set(p.reasons) & {REASON_SCANNED, REASON_INVISIBLE_LAYER, REASON_SPARSE_OVER_IMAGE} for p in ocr_pages
        )
        return (DOC_SCANNED, 0.95) if scan_like else (DOC_IMAGE_BASED, 0.8)
    share = len(ocr_pages) / len(content_pages)
    return DOC_MIXED, round(0.5 + 0.4 * (1 - abs(0.5 - share) * 2) if not sampled else 0.6, 3)


def inspect_pdf(
    source: Any,
    pages: Optional[Iterable[int]] = None,
    sample: Optional[int] = None,
    password: Optional[str] = None,
    config: Optional[InspectionConfig] = None,
) -> DocumentInspection:
    """Inspect a PDF path/bytes.

    Parameters
    ----------
    pages:
        1-indexed pages to inspect (default: all).
    sample:
        Inspect only ``sample`` evenly distributed pages (fast triage of
        huge documents); the document type is then an estimate.
    """
    from textlens.inputs.source import open_source

    src = open_source(source)
    if not src.is_pdf:
        from textlens.errors import UnsupportedInputError

        raise UnsupportedInputError(f"{src.name} is not a PDF ({src.mime}).")
    t0 = time.perf_counter()
    target = src.data if src.data is not None else src.path
    with open_pdf(target, password=password) as doc:
        with PDFIUM_LOCK:
            count = len(doc)
            if pages is not None:
                wanted = sorted({p for p in pages if 1 <= p <= count})
            elif sample:
                wanted = sample_pages(count, sample)
            else:
                wanted = list(range(1, count + 1))
            results = []
            for number in wanted:
                page = doc[number - 1]
                try:
                    results.append(analyze_page(page, number, config))
                finally:
                    page.close()
            metadata = {k: v for k, v in (doc.get_metadata_dict() or {}).items() if v}
    sampled = len(wanted) < count
    pdf_type, confidence = classify_document(results, count, sampled)
    ocr_pages = [p.number for p in results if p.needs_ocr]
    return DocumentInspection(
        page_count=count,
        pdf_type=pdf_type,
        confidence=confidence,
        pages=results,
        pages_needing_ocr=ocr_pages,
        reasons_by_page={p.number: p.reasons for p in results if p.reasons},
        sampled=sampled,
        metadata=metadata,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )
