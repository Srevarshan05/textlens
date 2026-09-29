"""
textlens.documents.document
───────────────────────────
A lazy, document-centric view over the TextLens pipeline.

    >>> import textlens
    >>> doc = textlens.load("paper.pdf")
    >>> doc.page_count                  # cheap, nothing processed yet
    >>> doc.page(3).text                # processes only page 3
    >>> doc.inspect().pdf_type          # classification, no OCR
    >>> doc.to_markdown()               # processes the rest on demand
    >>> for chunk in doc.chunks(max_tokens=400): ...

Pages already processed are reused, so mixing ``page(n)`` calls with
whole-document exports never OCRs a page twice.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from textlens.core.result import Chunk, Page, Result, Table


class Document:
    def __init__(self, source: Any, ocr: Any = None, **options: Any) -> None:
        from textlens.inputs.source import open_source

        self.source = open_source(source)
        if ocr is None:
            from textlens.core.engine import OCR

            ocr = OCR()
        self.ocr = ocr
        self.options = options
        self._pages: Dict[int, Page] = {}
        self._result: Optional[Result] = None
        self._page_count: Optional[int] = None

    # ── cheap facts ──────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self.source.name

    @property
    def kind(self) -> str:
        return self.source.kind

    @property
    def page_count(self) -> int:
        if self._page_count is None:
            if self.source.is_pdf:
                from textlens.documents.pdf.pdfium import PDFIUM_LOCK, open_pdf

                with open_pdf(self.source.data if self.source.data is not None else self.source.path, password=self.options.get("password")) as doc, PDFIUM_LOCK:
                    self._page_count = len(doc)
            elif self.source.is_image and self.source.image is None:
                from PIL import Image

                with Image.open(self.source.open_binary()) as im:
                    self._page_count = getattr(im, "n_frames", 1)
            else:
                self._page_count = len(self.result.pages) if self.source.kind in ("docx", "pptx") else 1
        return self._page_count

    def inspect(self) -> Any:
        return self.ocr.inspect(self.source, password=self.options.get("password"))

    def metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"name": self.name, "kind": self.kind, "mime": self.source.mime, "size_bytes": self.source.size, "sha256": self.source.sha256}
        if self.source.is_pdf:
            from textlens.documents.pdf.pdfium import PDFIUM_LOCK, open_pdf

            with open_pdf(self.source.data if self.source.data is not None else self.source.path, password=self.options.get("password")) as doc, PDFIUM_LOCK:
                meta.update({k.lower(): v for k, v in (doc.get_metadata_dict() or {}).items() if v})
                meta["page_count"] = len(doc)
        return meta

    # ── page access ──────────────────────────────────────────────────────
    def page(self, number: int) -> Page:
        """Process (once) and return a 1-indexed page."""
        if number not in self._pages:
            if self._result is not None:
                for p in self._result.pages:
                    self._pages[p.number] = p
            if number not in self._pages:
                for p in self.ocr.stream(self.source, pages=[number], **self.options):
                    self._pages[p.number] = p
        if number not in self._pages:
            raise IndexError(f"page {number} out of range (1..{self.page_count})")
        return self._pages[number]

    def iter_pages(self) -> Iterator[Page]:
        """Stream pages in order, reusing already-processed ones."""
        todo = [n for n in range(1, self.page_count + 1) if n not in self._pages] if self.source.is_pdf else None
        if todo is None:
            for p in self.ocr.stream(self.source, **self.options):
                self._pages[p.number] = p
                yield p
            return
        pending = iter(self.ocr.stream(self.source, pages=todo, **self.options)) if todo else iter(())
        for n in range(1, self.page_count + 1):
            if n not in self._pages:
                p = next(pending)
                self._pages[p.number] = p
            yield self._pages[n]

    @property
    def result(self) -> Result:
        """The full :class:`Result`, processing only pages not seen yet."""
        if self._result is None:
            import time

            from textlens.runtime.cache import config_hash

            t0 = time.perf_counter()
            pages = list(self.iter_pages())
            opts = self.ocr._options(dict(self.options))
            cfg = self.ocr._config(opts)
            self._result = self.ocr._assemble(self.source, pages, opts, cfg, config_hash(cfg), t0)
        return self._result

    def pages(self) -> List[Page]:
        return self.result.pages

    # ── content views ────────────────────────────────────────────────────
    def text(self) -> str:
        return self.result.text

    def tables(self) -> List[Table]:
        return self.result.tables

    def layout(self) -> List[Dict[str, Any]]:
        return self.result.layout

    def to_markdown(self, **kwargs: Any) -> str:
        return self.result.to_markdown(**kwargs)

    def to_json(self, **kwargs: Any) -> str:
        return self.result.to_json(**kwargs)

    def to_html(self, **kwargs: Any) -> str:
        return self.result.to_html(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return self.result.to_dict()

    def chunks(self, **kwargs: Any) -> List[Chunk]:
        return self.result.to_chunks(**kwargs)

    def images(self, min_side: int = 32) -> List[Dict[str, Any]]:
        """Embedded raster images (PDF) or the image itself, with positions."""
        if not self.source.is_pdf:
            from textlens.inputs.images import load_image

            img = self.source.image if self.source.image is not None else load_image(self.source.open_binary())
            return [{"page": 1, "bbox": [0, 0, img.size[0], img.size[1]], "image": img}]
        import pypdfium2.raw as raw

        from textlens.documents.pdf.pdfium import PDFIUM_LOCK, open_pdf, page_transform

        out: List[Dict[str, Any]] = []
        with open_pdf(self.source.data if self.source.data is not None else self.source.path, password=self.options.get("password")) as doc, PDFIUM_LOCK:
            for i in range(len(doc)):
                page = doc[i]
                xf = page_transform(page)
                for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=4):
                    try:
                        pil = obj.get_bitmap(render=False).to_pil()
                    except Exception:
                        continue
                    if min(pil.size) < min_side:
                        continue
                    out.append({"page": i + 1, "bbox": [round(v, 2) for v in xf.rect(*obj.get_bounds())], "image": pil, "pixels": list(pil.size)})
                page.close()
        return out

    def __len__(self) -> int:
        return self.page_count

    def __iter__(self) -> Iterator[Page]:
        return self.iter_pages()

    def __repr__(self) -> str:
        done = len(self._pages)
        return f"<Document {self.name!r} kind={self.kind} pages={self._page_count if self._page_count is not None else '?'} processed={done}>"


def load(source: Any, ocr: Any = None, **options: Any) -> Document:
    """Open a document lazily (see :class:`Document`)."""
    return Document(source, ocr=ocr, **options)
