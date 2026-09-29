"""
textlens.evaluation.visualize
─────────────────────────────
Visual debugging: draw what TextLens saw on every page.

Colours encode the content source — native text (green), OCR (blue),
generative VLM (purple), fused (orange), tables (red) — numbers give the
reading order, and low-confidence blocks get a dashed red outline.  A
legend strip shows the page's routing decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

SOURCE_COLORS = {"native": (39, 174, 96), "ocr": (41, 128, 185), "vlm": (142, 68, 173), "fused": (230, 126, 34), "empty": (127, 140, 141)}
TABLE_COLOR = (192, 57, 43)


def _page_images(source: Any, dpi: int, pages: Optional[List[int]]) -> List[tuple]:
    from textlens.inputs.images import iter_image_frames, to_rgb
    from textlens.inputs.source import open_source

    src = open_source(source)
    out = []
    if src.is_pdf:
        from textlens.documents.pdf.pdfium import PDFIUM_LOCK, open_pdf, render_page

        with open_pdf(src.data if src.data is not None else src.path) as doc, PDFIUM_LOCK:
            for i in range(len(doc)):
                if pages and (i + 1) not in pages:
                    continue
                page = doc[i]
                img, eff = render_page(page, dpi)
                page.close()
                out.append((i + 1, img, eff / 72.0))
    elif src.is_image:
        frames = [to_rgb(src.image)] if src.image is not None else list(iter_image_frames(src.open_binary()))
        for n, fr in enumerate(frames, start=1):
            if not pages or n in pages:
                out.append((n, fr, 1.0))
    return out


def draw_page(img: Any, page: Any, scale: float) -> Any:
    from PIL import Image, ImageDraw

    canvas = img.convert("RGB").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    review = 0.6
    for b in page.ordered_blocks():
        if b.bbox is None:
            continue
        x0, y0, x1, y1 = (v * scale for v in b.bbox)
        color = TABLE_COLOR if b.type == "table" else SOURCE_COLORS.get(b.source, (0, 0, 0))
        draw.rectangle([x0, y0, x1, y1], fill=color + (28,), outline=color + (255,), width=2)
        if b.confidence is not None and b.confidence < review:
            for x in range(int(x0), int(x1), 8):
                draw.line([x, y0 - 2, min(x + 4, x1), y0 - 2], fill=(231, 76, 60, 255), width=2)
        label = f"{b.order}" + (f" {b.type}" if b.type not in ("text",) else "")
        ly = y0 - 13 if y0 >= 13 else y1 + 1  # above the box, never over the text
        draw.rectangle([x0, ly, x0 + 7 * len(label) + 4, ly + 12], fill=color + (220,))
        draw.text((x0 + 2, ly), label, fill=(255, 255, 255, 255))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    prov = page.provenance
    legend = f"page {page.number} · {page.classification or '-'} · source={prov.source} · model={prov.model or '-'}"
    if prov.confidence is not None:
        legend += f" · conf={prov.confidence:.3f}"
    if prov.reasons:
        legend += f" · reasons={','.join(prov.reasons)}"
    strip = Image.new("RGB", (canvas.size[0], 22), (33, 33, 33))
    ImageDraw.Draw(strip).text((6, 5), legend[:200], fill=(255, 255, 255))
    out = Image.new("RGB", (canvas.size[0], canvas.size[1] + 22), (255, 255, 255))
    out.paste(strip, (0, 0))
    out.paste(canvas, (0, 22))
    return out


def render_overlays(source: Any, out_dir: Any, ocr: Any = None, pages: Optional[List[int]] = None, dpi: int = 110) -> List[Path]:
    """Run the pipeline and save one annotated PNG per page; returns paths."""
    if ocr is None:
        from textlens.core.engine import OCR

        ocr = OCR()
    result = ocr(source, pages=pages)
    by_number = {p.number: p for p in result.pages}
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for number, img, px_per_unit in _page_images(source, dpi, pages):
        page = by_number.get(number)
        if page is None:
            continue
        scale = px_per_unit if page.unit == "pt" else (img.size[0] / page.width if page.width else 1.0)
        path = dest / f"page-{number:04d}.png"
        draw_page(img, page, scale).save(path)
        written.append(path)
    return written
