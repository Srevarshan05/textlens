"""
textlens.core.pipeline
──────────────────────
Page-level orchestration: the heart of TextLens.

::

    Source ──► PDF? ──yes──► for each page (streamed, lazily):
      │                        inspect (ms, no rendering)
      │                        ├─ native text OK ──► positioned native extraction
      │                        │                     └─ text-quality check (garbled → OCR)
      │                        ├─ OCR layer + fast profile ──► reuse the layer
      │                        ├─ empty page ──► skip
      │                        └─ needs OCR ──► render ─► route ─► OCR ─► validate
      │                                                     └─ low confidence ─► fallback model
      └──no (image) ──► difficulty signals ─► route ─► OCR ─► validate ─► fallback

Every page records its provenance: source (native / ocr / vlm / fused),
model, backend, device, DPI, confidence, OCR reasons, the routing
explanation, all attempts and stage timings.  A page that fails is returned
with warnings instead of aborting the document (partial results).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence

from textlens.backends.base import PageOCR, RecognizeOptions
from textlens.core.quality import assess_text
from textlens.core.result import (
    SOURCE_EMPTY,
    SOURCE_FUSED,
    SOURCE_OCR,
    SOURCE_VLM,
    Attempt,
    BBox,
    Block,
    Page,
    PageProvenance,
)
from textlens.core.router import PageSignals, Router
from textlens.errors import InputTooLargeError, TextLensError

logger = logging.getLogger("textlens.pipeline")

OCR_MODES = ("auto", "force", "off")


@dataclass
class PipelineOptions:
    ocr: str = "auto"  # auto = selective OCR | force = OCR every page | off = native only
    pages: Optional[Sequence[int]] = None  # 1-indexed page selection
    task: str = "text"
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None
    detect_tables: bool = True
    ocr_images: bool = False  # OCR embedded images on native pages ("fused" pages)
    reuse_ocr_layer: Optional[bool] = None  # None → profile default
    password: Optional[str] = None
    keep_images: bool = False
    max_pages: Optional[int] = None
    events: Any = None  # optional callback(event: str, payload: dict)


def scale_page(page: Page, factor: float, unit: str) -> Page:
    """Scale every coordinate on a page (pixels → points etc.)."""

    def sb(b: Optional[BBox]) -> Optional[BBox]:
        return b.scale(factor).rounded(2) if b is not None else None

    for blk in page.blocks:
        blk.bbox = sb(blk.bbox)
        for ln in blk.lines:
            ln.bbox = sb(ln.bbox)
            if ln.polygon:
                ln.polygon = [(round(x * factor, 2), round(y * factor, 2)) for x, y in ln.polygon]
            for w in ln.words:
                w.bbox = sb(w.bbox)
    for t in page.tables:
        t.bbox = sb(t.bbox)
        for c in t.cells:
            c.bbox = sb(c.bbox)
    for f in page.formulas:
        f.bbox = sb(f.bbox)
    page.width = round(page.width * factor, 2)
    page.height = round(page.height * factor, 2)
    page.unit = unit
    return page


def _emit(options: PipelineOptions, event: str, **payload: Any) -> None:
    if options.events is not None:
        try:
            options.events(event, payload)
        except Exception:  # user callbacks must never break processing
            logger.debug("event callback failed", exc_info=True)


class Pipeline:
    def __init__(self, router: Router, options: Optional[PipelineOptions] = None) -> None:
        self.router = router
        self.options = options or PipelineOptions()
        if self.options.ocr not in OCR_MODES:
            from textlens.errors import ConfigurationError

            raise ConfigurationError(f"ocr must be one of {OCR_MODES}, got {self.options.ocr!r}")

    # ── entry point ──────────────────────────────────────────────────────
    def run(self, source: Any) -> Iterator[Page]:
        """Yield pages one at a time (constant memory for large PDFs)."""
        if source.is_pdf:
            yield from self._pdf_pages(source)
        elif source.is_image:
            yield from self._image_pages(source)
        else:
            from textlens.documents.office import office_pages

            yield from office_pages(source, self)

    # ── images ───────────────────────────────────────────────────────────
    def _image_pages(self, source: Any) -> Iterator[Page]:
        from textlens.core.difficulty import analyze_image
        from textlens.inputs.images import iter_image_frames, to_rgb

        frames = [to_rgb(source.image)] if source.image is not None else iter_image_frames(source.open_binary())
        wanted = set(self.options.pages or [])
        for number, frame in enumerate(frames, start=1):
            if wanted and number not in wanted:
                continue
            if self.options.ocr == "off":
                yield Page(number=number, width=frame.size[0], height=frame.size[1], unit="px", classification="image",
                           provenance=PageProvenance(page=number, source=SOURCE_EMPTY, warnings=["ocr='off' on an image input"]))
                continue
            t0 = time.perf_counter()
            sig = analyze_image(frame)
            signals = PageSignals(kind="image", likely_table=sig.likely_table, difficulty=sig.difficulty, task=self.options.task)
            page = self.ocr_image(frame, number, signals)
            page.classification = "image"
            page.provenance.timings_ms["analyze"] = round((time.perf_counter() - t0) * 1000 - sum(page.provenance.timings_ms.values()), 1)
            if self.options.keep_images:
                page.image = frame  # type: ignore[attr-defined]
            yield page

    # ── PDFs ─────────────────────────────────────────────────────────────
    def _pdf_pages(self, source: Any) -> Iterator[Page]:
        from textlens.config import get_settings
        from textlens.documents.pdf.inspector import KIND_EMPTY, KIND_GRAPHICS, REASON_GARBLED, analyze_page
        from textlens.documents.pdf.native import extract_page
        from textlens.documents.pdf.pdfium import PDFIUM_LOCK, open_pdf, render_page

        opts = self.options
        profile = self.router.profile
        reuse_layer = profile.reuse_ocr_layer if opts.reuse_ocr_layer is None else opts.reuse_ocr_layer
        target = source.data if source.data is not None else source.path
        max_pages = opts.max_pages or get_settings().max_pages
        with open_pdf(target, password=opts.password) as doc:
            with PDFIUM_LOCK:
                count = len(doc)
            if count > max_pages and not opts.pages:
                raise InputTooLargeError(
                    f"{source.name} has {count} pages (limit {max_pages}).",
                    hint="Pass pages=[…] or raise TEXTLENS_MAX_PAGES.",
                )
            numbers = [p for p in (opts.pages or range(1, count + 1)) if 1 <= p <= count]
            _emit(opts, "document_start", page_count=count, pages=len(numbers))
            for number in numbers:
                t0 = time.perf_counter()
                image = None
                try:
                    with PDFIUM_LOCK:
                        pdf_page = doc[number - 1]
                        try:
                            insp = analyze_page(pdf_page, number)
                            t_inspect = (time.perf_counter() - t0) * 1000
                            needs_ocr = insp.needs_ocr
                            native: Optional[Page] = None
                            reasons = list(insp.reasons)
                            if opts.ocr == "force":
                                needs_ocr = True
                                reasons = reasons or ["forced"]
                            elif opts.ocr == "off":
                                needs_ocr = False
                            if not needs_ocr and insp.kind not in (KIND_EMPTY, KIND_GRAPHICS):
                                native = extract_page(pdf_page, number, include_invisible=False, detect_tables=opts.detect_tables)
                                q = assess_text(native.text)
                                if q.garbled and opts.ocr == "auto":
                                    needs_ocr, native = True, None
                                    reasons = [REASON_GARBLED]
                            elif needs_ocr and insp.has_ocr_layer and reuse_layer and opts.ocr == "auto":
                                native = extract_page(pdf_page, number, include_invisible=True, detect_tables=opts.detect_tables)
                                if assess_text(native.text).garbled or not native.text.strip():
                                    native = None
                                else:
                                    needs_ocr = False
                                    native.provenance.warnings.append("reused the document's existing OCR text layer")
                                    native.provenance.reasons = list(insp.reasons)
                            if needs_ocr or (opts.ocr_images and insp.kind == "mixed" and native is not None):
                                image, eff_dpi = render_page(pdf_page, self.router.dpi or profile.dpi)
                        finally:
                            pdf_page.close()
                    t_native = (time.perf_counter() - t0) * 1000 - t_inspect
                    if needs_ocr and image is not None:
                        signals = PageSignals(
                            kind=insp.kind,
                            reasons=reasons,
                            likely_table=insp.likely_table,
                            likely_formulas=insp.likely_formulas,
                            difficulty=self._pdf_difficulty(image, insp),
                            has_ocr_layer=insp.has_ocr_layer,
                            task=opts.task,
                        )
                        page = self.ocr_image(image, number, signals, points_per_pixel=72.0 / eff_dpi)
                        page.provenance.dpi = round(eff_dpi, 1)
                        page.provenance.reasons = reasons
                        page.provenance.timings_ms["render"] = round(t_native, 1)
                    elif native is not None:
                        page = native
                        if opts.ocr_images and image is not None:
                            self._fuse_image_regions(page, image, 72.0 / eff_dpi, insp)
                        page.provenance.timings_ms["extract"] = round(t_native, 1)
                    else:
                        page = Page(number=number, width=insp.width, height=insp.height, unit="pt",
                                    provenance=PageProvenance(page=number, source=SOURCE_EMPTY))
                    page.classification = insp.kind
                    page.rotation = insp.rotation
                    page.provenance.timings_ms["inspect"] = round(t_inspect, 2)
                    if self.options.keep_images and image is not None:
                        page.image = image  # type: ignore[attr-defined]
                except TextLensError as exc:
                    if isinstance(exc, InputTooLargeError):
                        raise
                    logger.warning("Page %d failed: %s", number, exc)
                    page = Page(number=number, unit="pt", provenance=PageProvenance(page=number, source=SOURCE_EMPTY, warnings=[f"failed: {exc.message}"], needs_review=True))
                _emit(opts, "page_done", page=number, source=page.provenance.source, model=page.provenance.model)
                yield page

    @staticmethod
    def _pdf_difficulty(image: Any, insp: Any) -> float:
        from textlens.core.difficulty import analyze_image

        d = analyze_image(image).difficulty
        if insp.likely_table:
            d = max(d, 0.5)
        if insp.image_dpi and insp.image_dpi < 150:
            d = min(1.0, d + 0.15)  # low-resolution scan
        return round(d, 3)

    # ── OCR with validation and fallback ─────────────────────────────────
    def ocr_image(self, image: Any, number: int, signals: PageSignals, points_per_pixel: Optional[float] = None) -> Page:
        from textlens.backends.loader import get_backend

        decision = self.router.route(signals)
        chain = [decision.model] + list(decision.fallbacks)
        attempts: List[Attempt] = []
        best: Optional[Page] = None
        best_key: tuple = (-1, -1.0)
        last_error: Optional[TextLensError] = None
        for model_id in chain:
            spec = self.router.spec_for(model_id)
            device, backend_opts = self.router.options_for(model_id)
            t0 = time.perf_counter()
            try:
                backend = get_backend(spec, device, **backend_opts)
                result: PageOCR = backend.recognize(
                    [image],
                    RecognizeOptions(task=signals.task, prompt=self.options.prompt, max_new_tokens=self.options.max_new_tokens),
                )[0]
            except TextLensError as exc:
                last_error = exc
                attempts.append(Attempt(model=model_id, backend=spec.backend, accepted=False, reason=f"error: {exc.message}", elapsed_ms=round((time.perf_counter() - t0) * 1000, 1)))
                _emit(self.options, "attempt_failed", page=number, model=model_id, error=exc.message)
                continue
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            page = self._to_page(result, number)
            quality = assess_text(page.text) if page.text else None
            conf = result.confidence
            empty = not page.text.strip()
            garbled = bool(quality and quality.garbled)
            confident = conf is None or conf >= decision.accept_confidence
            accepted = not empty and not garbled and confident
            why = "accepted" if accepted else "empty output" if empty else "garbled output" if garbled else f"confidence {conf:.3f} < {decision.accept_confidence:.2f}"
            attempts.append(Attempt(model=model_id, backend=result.backend, confidence=conf, accepted=accepted, reason=why, elapsed_ms=elapsed))
            # Rank: usable text first, then confidence (unscored VLM output
            # counts as meeting the bar — it only runs as a quality fallback).
            key = (0 if (empty or garbled) else 1, conf if conf is not None else decision.accept_confidence)
            if best is None or key > best_key:
                best, best_key = page, key
            if accepted or not decision.fallback_enabled:
                break
            _emit(self.options, "fallback", page=number, model=model_id, reason=why)
        if best is None:
            assert last_error is not None
            raise last_error
        prov = best.provenance
        prov.attempts = attempts
        prov.routing = decision.explain()
        prov.reasons = list(signals.reasons)
        winner = next((a for a in attempts if a.model == prov.model and a.reason != "" and not a.reason.startswith("error")), None)
        prov.confidence = best.confidence if best.confidence is not None else (winner.confidence if winner else None)
        prov.needs_review = (prov.confidence is not None and prov.confidence < decision.review_confidence) or not best.text.strip()
        if points_per_pixel:
            scale_page(best, points_per_pixel, "pt")
        return best

    def _to_page(self, result: PageOCR, number: int) -> Page:
        from textlens.core.markdown import parse_markdown
        from textlens.documents.layout import build_blocks, finalize_order

        prov = PageProvenance(
            page=number,
            model=result.model,
            model_revision=result.model_revision,
            backend=result.backend,
            device=result.device,
            confidence=result.confidence,
            timings_ms=dict(result.timings_ms),
            warnings=list(result.warnings),
        )
        if result.markdown is not None:
            blocks, tables, formulas = parse_markdown(result.markdown, page=number, model=result.model, backend=result.backend, confidence=result.confidence)
            prov.source = SOURCE_VLM if blocks else SOURCE_EMPTY
            return Page(number=number, width=result.width, height=result.height, unit="px", blocks=blocks, tables=tables,
                        formulas=formulas, markdown=result.markdown, provenance=prov)
        blocks = build_blocks(result.items, number, result.width, result.height, SOURCE_OCR, unit="line", model=result.model, backend=result.backend)
        finalize_order(blocks, number)
        prov.source = SOURCE_OCR if blocks else SOURCE_EMPTY
        return Page(number=number, width=result.width, height=result.height, unit="px", blocks=blocks, provenance=prov)

    def _fuse_image_regions(self, page: Page, image: Any, points_per_pixel: float, insp: Any) -> None:
        """OCR large embedded images (screenshots, figures) on a native page."""
        from textlens.documents.layout import finalize_order

        regions = [b for b in (insp.image_boxes or []) if (b[2] - b[0]) * (b[3] - b[1]) >= 0.02 * insp.width * insp.height]
        if not regions:
            return
        added: List[Block] = []
        px_per_pt = 1.0 / points_per_pixel
        for x0, y0, x1, y1 in regions[:8]:
            box_px = (int(x0 * px_per_pt), int(y0 * px_per_pt), int(x1 * px_per_pt), int(y1 * px_per_pt))
            crop = image.crop(box_px)
            if min(crop.size) < 24:
                continue
            sub = self.ocr_image(crop, page.number, PageSignals(kind="figure", task="text"))
            for blk in sub.blocks:
                if blk.bbox is not None:
                    blk.bbox = BBox(
                        blk.bbox.x0 * points_per_pixel + x0, blk.bbox.y0 * points_per_pixel + y0,
                        blk.bbox.x1 * points_per_pixel + x0, blk.bbox.y1 * points_per_pixel + y0,
                    ).rounded(2)
                    for ln in blk.lines:
                        if ln.bbox is not None:
                            ln.bbox = BBox(ln.bbox.x0 * points_per_pixel + x0, ln.bbox.y0 * points_per_pixel + y0,
                                           ln.bbox.x1 * points_per_pixel + x0, ln.bbox.y1 * points_per_pixel + y0).rounded(2)
                        ln.words = []
                blk.attributes["region"] = "embedded_image"
                added.append(blk)
        if added:
            merged = sorted(page.blocks + added, key=lambda b: (b.bbox.y0 if b.bbox else 0, b.bbox.x0 if b.bbox else 0))
            page.blocks = finalize_order(merged, page.number)
            page.provenance.source = SOURCE_FUSED
            page.provenance.warnings.append(f"OCR'd {len(added)} block(s) inside embedded images")
