"""
Deterministic PDF fixtures for tests — no third-party PDF writer needed.

Each helper returns *page specs*; :func:`build_pdf` assembles them into a
valid PDF byte string with a correct cross-reference table.  Scanned pages
embed a JPEG of genuinely rendered text so OCR assertions are meaningful.
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

PAGE_W, PAGE_H = 612, 792  # US Letter, points


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def text_image(lines: Sequence[str], width: int = 1275, height: int = 1650, size: int = 34, noise: bool = False) -> Image.Image:
    """Render lines of text onto a white page-sized image (≈150 dpi Letter)."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(size)
    y = 120
    for line in lines:
        draw.text((110, y), line, fill="black", font=font)
        y += int(size * 1.8)
    if noise:
        import random

        rnd = random.Random(7)
        for _ in range(2500):
            x, yy = rnd.randrange(width), rnd.randrange(height)
            img.putpixel((x, yy), (rnd.randrange(150, 255),) * 3)
    return img


def jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@dataclass
class PageSpec:
    content: str = ""
    images: Dict[str, Tuple[bytes, int, int]] = field(default_factory=dict)  # name -> (jpeg, w, h)
    forms: Dict[str, Tuple[str, Dict[str, Tuple[bytes, int, int]]]] = field(default_factory=dict)
    rotate: int = 0
    garbled_font: bool = False


def native_text_page(lines: Sequence[str], size: int = 12, x: float = 72, top: float = 720, leading: float = 16) -> PageSpec:
    ops = ["BT", f"/F1 {size} Tf", f"{leading} TL", f"{x} {top} Td"]
    for line in lines:
        ops.append(f"({_esc(line)}) Tj T*")
    ops.append("ET")
    return PageSpec(content="\n".join(ops))


def heading_and_body_page(title: str, heading: str, body: Sequence[str]) -> PageSpec:
    ops = ["BT", "/F2 24 Tf", f"72 720 Td ({_esc(title)}) Tj", "ET"]
    ops += ["BT", "/F2 16 Tf", f"72 680 Td ({_esc(heading)}) Tj", "ET"]
    ops += ["BT", "/F1 11 Tf", "14 TL", "72 650 Td"]
    ops += [f"({_esc(line)}) Tj T*" for line in body]
    ops.append("ET")
    return PageSpec(content="\n".join(ops))


def two_column_page(left: Sequence[str], right: Sequence[str], title: Optional[str] = None) -> PageSpec:
    ops: List[str] = []
    if title:
        ops += ["BT", "/F2 20 Tf", f"72 740 Td ({_esc(title)}) Tj", "ET"]
    for x, lines in ((72, left), (330, right)):
        ops += ["BT", "/F1 10 Tf", "13 TL", f"{x} 700 Td"]
        ops += [f"({_esc(line)}) Tj T*" for line in lines]
        ops.append("ET")
    return PageSpec(content="\n".join(ops))


def scanned_page(lines: Sequence[str], noise: bool = False) -> PageSpec:
    img = text_image(lines, noise=noise)
    data = jpeg_bytes(img)
    return PageSpec(content=f"q {PAGE_W} 0 0 {PAGE_H} 0 0 cm /Im1 Do Q", images={"Im1": (data, *img.size)})


def scanned_with_ocr_layer(lines: Sequence[str]) -> PageSpec:
    """Scan image with an invisible (render mode 3) text layer on top."""
    spec = scanned_page(lines)
    layer = ["BT", "3 Tr", "/F1 12 Tf", "16 TL", "72 720 Td"]
    layer += [f"({_esc(line)}) Tj T*" for line in lines]
    layer.append("ET")
    spec.content += "\n" + "\n".join(layer)
    return spec


def form_wrapped_scan(lines: Sequence[str]) -> PageSpec:
    img = text_image(lines)
    data = jpeg_bytes(img)
    form_content = f"q {PAGE_W} 0 0 {PAGE_H} 0 0 cm /Im1 Do Q"
    return PageSpec(content="q /Fm1 Do Q", forms={"Fm1": (form_content, {"Im1": (data, *img.size)})})


def vector_text_page(strokes: int = 2500) -> PageSpec:
    """Many tiny path segments and no text operators (outlined glyphs)."""
    ops = ["0 0 0 RG 0.5 w"]
    for i in range(strokes):
        x = 72 + (i % 90) * 5
        y = 700 - (i // 90) * 9
        ops.append(f"{x} {y} m {x + 3} {y + 6} l {x + 4} {y} l S")
    return PageSpec(content="\n".join(ops))


def ruled_table_page(rows: Sequence[Sequence[str]], x0: float = 72, y_top: float = 700, col_w: float = 150, row_h: float = 24) -> PageSpec:
    n_rows, n_cols = len(rows), len(rows[0])
    ops = ["0 0 0 RG 1 w"]
    for r in range(n_rows + 1):
        y = y_top - r * row_h
        ops.append(f"{x0} {y} m {x0 + n_cols * col_w} {y} l S")
    for c in range(n_cols + 1):
        x = x0 + c * col_w
        ops.append(f"{x} {y_top} m {x} {y_top - n_rows * row_h} l S")
    ops += ["BT", "/F1 11 Tf"]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ops.append(f"1 0 0 1 {x0 + c * col_w + 6} {y_top - (r + 1) * row_h + 8} Tm ({_esc(cell)}) Tj")
    ops.append("ET")
    return PageSpec(content="\n".join(ops))


def garbled_text_page(lines: Sequence[str]) -> PageSpec:
    """Readable glyphs whose ToUnicode map is a shifted alphabet."""
    ops = ["BT", "/F3 12 Tf", "16 TL", "72 720 Td"]
    ops += [f"({_esc(line)}) Tj T*" for line in lines]
    ops.append("ET")
    return PageSpec(content="\n".join(ops), garbled_font=True)


def blank_page() -> PageSpec:
    return PageSpec(content="")


def _shifted_tounicode() -> bytes:
    """CMap that permutes letters across both cases.

    Mirrors real broken ToUnicode maps (``Certificate`` → ``8VceZWZTReV``):
    words flip case mid-word and the letter histogram is permuted.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    entries = []
    for code in range(0x20, 0x7F):
        ch = chr(code)
        if ch.isalpha():
            shifted = alphabet[(alphabet.index(ch) * 7 + 3) % 52]
        else:
            shifted = ch
        entries.append(f"<{code:02X}> <{ord(shifted):04X}>")
    body = "\n".join(entries)
    cmap = (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Garbled def\n/CMapType 2 def\n1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        f"{len(entries)} beginbfchar\n{body}\nendbfchar\nendcmap\n"
        "CMapName currentdict /CMap defineresource pop\nend\nend\n"
    )
    return cmap.encode("latin-1")


def build_pdf(pages: Sequence[PageSpec], compress: bool = True, info: Optional[Dict[str, str]] = None) -> bytes:
    objects: List[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    def stream(data: bytes, extra: str = "") -> bytes:
        if compress:
            data = zlib.compress(data)
            extra += " /Filter /FlateDecode"
        return f"<< /Length {len(data)}{extra} >>\nstream\n".encode() + data + b"\nendstream"

    catalog = add(b"")  # placeholder
    pages_obj = add(b"")
    f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    tounicode = add(stream(_shifted_tounicode()))
    f3 = add(
        f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode {tounicode} 0 R >>".encode()
    )
    page_ids: List[int] = []
    for spec in pages:
        xobjs: Dict[str, int] = {}
        for name, (data, w, h) in spec.images.items():
            xobjs[name] = add(
                f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
                f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\nstream\n".encode()
                + data
                + b"\nendstream"
            )
        for name, (content, imgs) in spec.forms.items():
            inner = {}
            for iname, (data, w, h) in imgs.items():
                inner[iname] = add(
                    f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
                    f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\nstream\n".encode()
                    + data
                    + b"\nendstream"
                )
            res = " ".join(f"/{k} {v} 0 R" for k, v in inner.items())
            xobjs[name] = add(
                stream(
                    content.encode("latin-1"),
                    f" /Type /XObject /Subtype /Form /BBox [0 0 {PAGE_W} {PAGE_H}] /Resources << /XObject << {res} >> >>",
                )
            )
        content_id = add(stream(spec.content.encode("latin-1")))
        xres = " ".join(f"/{k} {v} 0 R" for k, v in xobjs.items())
        rot = f" /Rotate {spec.rotate}" if spec.rotate else ""
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}]{rot} "
                f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R /F3 {f3} 0 R >> /XObject << {xres} >> >> "
                f"/Contents {content_id} 0 R >>".encode()
            )
        )
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects[pages_obj - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    info_id = None
    if info:
        entries = " ".join(f"/{k} ({_esc(v)})" for k, v in info.items())
        info_id = add(f"<< {entries} >>".encode())

    out = io.BytesIO()
    out.write(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    trailer = f"<< /Size {len(objects) + 1} /Root {catalog} 0 R" + (f" /Info {info_id} 0 R" if info_id else "") + " >>"
    out.write(f"trailer\n{trailer}\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


LOREM = [
    "TextLens extracts native PDF text without running OCR.",
    "Selective OCR only processes pages that actually need it.",
    "Every block keeps its page number, bounding box and source.",
    "This sentence exists to give the page realistic text density.",
    "Invoices, contracts and research papers are common inputs.",
]
