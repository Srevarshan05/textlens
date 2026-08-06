"""
textlens.utils.image_utils
────────────────────────────
Unified Image & PDF Input Preprocessor for TextLens.

Supports loading and converting:
- Single & multi-page PDF files (via pypdfium2)
- Local image paths (.png, .jpg, .webp, .bmp, .tiff)
- HTTP/HTTPS URLs (images & PDFs)
- PIL.Image.Image instances
- Raw bytes / BytesIO buffers

Handles page selection, high-DPI PDF rendering, and format normalization.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("textlens.utils.image_utils")

try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    _PIL = False

try:
    import pypdfium2 as pdfium
    _PDFIUM = True
except ImportError:
    _PDFIUM = False

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


def render_pdf_to_images(
    pdf_source: Union[str, Path, bytes, io.BytesIO],
    dpi: int = 200,
    page_selection: Optional[Union[int, List[int]]] = None,
) -> List[PILImage.Image]:
    """Render page(s) of a PDF document into a list of PIL Images.

    Parameters
    ----------
    pdf_source : str | Path | bytes | io.BytesIO
        PDF file path, byte string, or BytesIO buffer.
    dpi : int, optional
        Rendering DPI resolution. Defaults to ``200``.
    page_selection : int | list[int], optional
        Page(s) to render. If ``None`` (default), all pages are rendered.
        Supports 1-indexed page numbers (e.g. ``1`` for first page) and 0-indexed.

    Returns
    -------
    list[PIL.Image.Image]
        Rendered RGB PIL images for each requested page.
    """
    if not _PDFIUM:
        raise ImportError(
            "pypdfium2 is required for PDF parsing in TextLens.\n"
            "Run: pip install pypdfium2"
        )
    if not _PIL:
        raise ImportError("Pillow is required. Run: pip install pillow")

    # Initialize pdfium document
    if isinstance(pdf_source, (str, Path)):
        p = Path(str(pdf_source))
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {p}")
        pdf = pdfium.PdfDocument(str(p))
    elif isinstance(pdf_source, (bytes, bytearray)):
        pdf = pdfium.PdfDocument(bytes(pdf_source))
    elif isinstance(pdf_source, io.BytesIO):
        pdf = pdfium.PdfDocument(pdf_source.getvalue())
    else:
        raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

    total_pages = len(pdf)
    if total_pages == 0:
        raise ValueError("PDF file contains 0 pages.")

    # Resolve target page indices (0-indexed internally)
    target_indices: List[int] = []
    if page_selection is None:
        target_indices = list(range(total_pages))
    elif isinstance(page_selection, int):
        idx = page_selection
        # Handle 1-indexed conversion if positive
        if idx > 0 and idx <= total_pages:
            idx = idx - 1
        elif idx < 0:
            idx = total_pages + idx
        if not (0 <= idx < total_pages):
            raise IndexError(f"Page index {page_selection} out of range for PDF with {total_pages} pages.")
        target_indices = [idx]
    elif isinstance(page_selection, (list, tuple)):
        for p_item in page_selection:
            idx = p_item - 1 if p_item > 0 else p_item
            if 0 <= idx < total_pages:
                target_indices.append(idx)
            else:
                logger.warning("Ignoring out-of-range page index %d (total pages: %d)", p_item, total_pages)

    if not target_indices:
        target_indices = list(range(total_pages))

    scale = dpi / 72.0
    images: List[PILImage.Image] = []

    for i in target_indices:
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil().convert("RGB")
        images.append(pil_img)

    logger.debug("Rendered %d page(s) from PDF at %d DPI", len(images), dpi)
    return images


def load_input_images(
    source: Any,
    dpi: int = 200,
    page: Optional[Union[int, List[int]]] = None,
) -> List[PILImage.Image]:
    """Unified loader that converts any input source into a list of RGB PIL Images.

    Supports:
    - ``PIL.Image.Image``
    - Local image files (.png, .jpg, .webp, .bmp, .tiff)
    - Local PDF files (.pdf)
    - URLs (http://, https://) pointing to images or PDFs
    - Raw bytes / BytesIO buffers (image or PDF)
    - Lists / tuples of any of the above

    Parameters
    ----------
    source : Any
        Input image or PDF source.
    dpi : int, optional
        Rendering DPI for PDFs. Defaults to ``200``.
    page : int | list[int], optional
        Page number(s) to render if *source* is a PDF.

    Returns
    -------
    list[PIL.Image.Image]
        List of RGB PIL Images ready for model input.
    """
    if not _PIL:
        raise ImportError("Pillow is required. Run: pip install pillow")

    # Handle sequence of sources
    if isinstance(source, (list, tuple)):
        result: List[PILImage.Image] = []
        for item in source:
            result.extend(load_input_images(item, dpi=dpi, page=page))
        return result

    # Handle PIL Image directly
    if isinstance(source, PILImage.Image):
        return [source.convert("RGB")]

    # Handle str / Path (local path or URL)
    if isinstance(source, (str, Path)):
        src_str = str(source).strip()

        # Handle URL
        if src_str.startswith(("http://", "https://")):
            if not _REQUESTS:
                raise ImportError("requests is required to fetch URLs. Run: pip install requests")
            resp = requests.get(src_str, timeout=30)
            resp.raise_for_status()
            content = resp.content

            if src_str.lower().endswith(".pdf") or content.startswith(b"%PDF"):
                return render_pdf_to_images(content, dpi=dpi, page_selection=page)
            else:
                img = PILImage.open(io.BytesIO(content)).convert("RGB")
                return [img]

        # Handle local path
        p = Path(src_str)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

        if p.suffix.lower() == ".pdf":
            return render_pdf_to_images(p, dpi=dpi, page_selection=page)
        else:
            img = PILImage.open(p).convert("RGB")
            return [img]

    # Handle bytes / BytesIO
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        if data.startswith(b"%PDF"):
            return render_pdf_to_images(data, dpi=dpi, page_selection=page)
        else:
            img = PILImage.open(io.BytesIO(data)).convert("RGB")
            return [img]

    if isinstance(source, io.BytesIO):
        source.seek(0)
        data = source.read()
        if data.startswith(b"%PDF"):
            return render_pdf_to_images(data, dpi=dpi, page_selection=page)
        else:
            img = PILImage.open(io.BytesIO(data)).convert("RGB")
            return [img]

    raise ValueError(f"Unsupported input source type: {type(source)}")
