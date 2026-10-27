"""PDF inspection (classification + OCR reasons), native extraction and layout."""

from __future__ import annotations

import pdf_factory as F
import pytest

from textlens.core.result import BBox
from textlens.documents.layout import Item, build_blocks, group_lines, mark_repeated_furniture, xy_cut
from textlens.documents.pdf import inspect_pdf
from textlens.documents.pdf.inspector import sample_pages
from textlens.documents.pdf.native import extract_page
from textlens.documents.pdf.pdfium import open_pdf


def test_page_classification_and_reasons(fixture_pdf):
    report = inspect_pdf(fixture_pdf)
    kinds = {p.number: (p.kind, p.needs_ocr, p.reasons) for p in report.pages}
    assert kinds[1] == ("native", False, [])
    assert kinds[2] == ("scanned", True, ["scanned"])
    assert kinds[3] == ("scanned", True, ["invisible_text_layer"])
    assert kinds[4] == ("broken_encoding", True, ["suspected_garbled_text"])
    assert kinds[5][0] == "empty" and kinds[5][1] is False
    assert kinds[6] == ("native", False, [])
    assert report.pdf_type == "mixed"
    assert report.pages_needing_ocr == [2, 3, 4]
    assert report.metadata.get("Title") == "Fixture Report"
    assert report.pages[5].likely_table
    assert report.pages[2].has_ocr_layer


def test_document_types():
    native = F.build_pdf([F.native_text_page(F.LOREM)] * 3)
    scanned = F.build_pdf([F.scanned_page(["a scan"]), F.form_wrapped_scan(["wrapped"])])
    vector = F.build_pdf([F.vector_text_page()])
    assert inspect_pdf(native).pdf_type == "text_based"
    s = inspect_pdf(scanned)
    assert s.pdf_type == "scanned" and s.pages_needing_ocr == [1, 2]
    v = inspect_pdf(vector)
    assert v.pdf_type == "image_based" and v.pages[0].reasons == ["vector_text"]
    assert inspect_pdf(F.build_pdf([F.blank_page()])).pdf_type == "empty"


def test_sampling_for_large_documents():
    assert sample_pages(100, 8)[0] == 1 and sample_pages(100, 8)[-1] == 100 and len(sample_pages(100, 8)) == 8
    assert sample_pages(3, 8) == [1, 2, 3]
    big = F.build_pdf([F.native_text_page(["page"] * 3)] * 20)
    r = inspect_pdf(big, sample=4)
    assert r.sampled and len(r.pages) == 4 and r.page_count == 20


def test_inspection_is_fast(fixture_pdf):
    r = inspect_pdf(fixture_pdf)
    assert all(p.elapsed_ms < 500 for p in r.pages)
    assert r.to_dict()["pages"][0]["likely_table"] is False


def test_image_boxes_and_rotation():
    rotated = F.native_text_page(F.LOREM)
    rotated.rotate = 90
    r = inspect_pdf(F.build_pdf([F.scanned_page(["x"]), rotated]))
    assert r.pages[0].image_boxes and r.pages[0].image_boxes[0][2] == pytest.approx(612, abs=1)
    assert (r.pages[1].width, r.pages[1].height) == (792.0, 612.0)  # display size after /Rotate 90


def _extract(pdf: bytes, index: int = 0, **kw):
    with open_pdf(pdf) as doc:
        return extract_page(doc[index], index + 1, **kw)


def test_native_extraction_types_and_coordinates():
    page = _extract(F.build_pdf([F.heading_and_body_page("Annual Report", "Introduction", F.LOREM)]))
    blocks = page.ordered_blocks()
    assert [b.type for b in blocks[:3]] == ["title", "heading", "text"]
    assert blocks[0].text == "Annual Report" and blocks[0].level == 1
    assert page.unit == "pt" and page.provenance.source == "native"
    for b in blocks:  # top-left origin, inside the page
        assert 0 <= b.bbox.x0 < b.bbox.x1 <= 612 and 0 <= b.bbox.y0 < b.bbox.y1 <= 792
    assert blocks[0].bbox.y0 < blocks[1].bbox.y0 < blocks[2].bbox.y0
    assert blocks[2].lines[0].words[0].bbox is not None


def test_two_column_reading_order():
    left = [f"Left column sentence number {i} continues here." for i in range(10)]
    right = [f"Right column sentence number {i} continues here." for i in range(10)]
    page = _extract(F.build_pdf([F.two_column_page(left, right, title="Columns")]))
    text = page.text
    assert text.index("Left column sentence number 9") < text.index("Right column sentence number 0")
    assert page.ordered_blocks()[0].text == "Columns"


def test_ruled_table_reconstruction():
    page = _extract(F.build_pdf([F.ruled_table_page([["Item", "Qty", "Price"], ["Apple", "3", "1.20"], ["Pear", "5", "0.80"]])]))
    assert len(page.tables) == 1
    assert page.tables[0].grid() == [["Item", "Qty", "Price"], ["Apple", "3", "1.20"], ["Pear", "5", "0.80"]]
    assert page.tables[0].cells[0].is_header
    assert [b.type for b in page.blocks] == ["table"]


def test_invisible_layer_only_read_on_request():
    pdf = F.build_pdf([F.scanned_with_ocr_layer(F.LOREM)])
    assert _extract(pdf).text == ""
    assert "Selective OCR" in _extract(pdf, include_invisible=True).text


# ── layout primitives ────────────────────────────────────────────────────────


def _words(rows, x0=0.0, gap=5.0, w=40.0, h=10.0, y0=0.0):
    items = []
    for r, row in enumerate(rows):
        x = x0
        for word in row:
            items.append(Item(word, BBox(x, y0 + r * 14, x + w, y0 + r * 14 + h), font_size=h))
            x += w + gap
    return items


def test_xy_cut_reads_prose_columns_left_then_right():
    left = _words([["alpha", "beta", "gamma", "delta"]] * 5)
    right = _words([["one", "two", "three", "four"]] * 5, x0=300)
    items = left + right
    regions = xy_cut(items)
    order = [items[i].text for region in regions for i in region]
    assert order.index("delta") < order.index("one")
    assert len(regions) == 2


def test_xy_cut_keeps_tabular_rows_together():
    # label ...... value rows (invoice-like): must stay row-major
    items = []
    for r, (label, value) in enumerate([("Design", "$10"), ("Consulting", "$20"), ("Support", "$30")]):
        items.append(Item(label, BBox(0, r * 20, 80, r * 20 + 10), font_size=10))
        items.append(Item(value, BBox(400, r * 20, 440, r * 20 + 10), font_size=10))
    blocks = build_blocks(items, 1, 612, 792, "ocr", unit="line")
    text = "\n".join(b.text for b in blocks)
    assert text.index("Design") < text.index("$10") < text.index("Consulting")


def test_group_lines_sorts_words():
    items = [Item("world", BBox(60, 0, 100, 10)), Item("hello", BBox(0, 1, 50, 11)), Item("next", BBox(0, 20, 30, 30))]
    lines = group_lines(items, range(3))
    assert [[items[i].text for i in ln] for ln in lines] == [["hello", "world"], ["next"]]


def test_repeated_headers_marked_as_furniture():
    from textlens.core.result import Block

    pages = []
    for n in range(4):
        pages.append([
            Block("text", "ACME Corp confidential", BBox(50, 10, 300, 20), page=n + 1),
            Block("text", f"Body {n}", BBox(50, 200, 300, 220), page=n + 1),
            Block("text", f"Page {n + 1}", BBox(280, 770, 330, 780), page=n + 1),
        ])
    mark_repeated_furniture(pages, [792.0] * 4)
    assert all(p[0].type == "page_header" and p[2].type == "page_footer" and p[1].type == "text" for p in pages)
