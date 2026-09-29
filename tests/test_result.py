"""Unified result schema: views, (de)serialisation, exporters, chunking."""

from __future__ import annotations

import json

import pytest

from textlens.core.result import (
    HEADING,
    LIST_ITEM,
    PAGE_FOOTER,
    TABLE,
    TEXT,
    TITLE,
    Attempt,
    BBox,
    Block,
    Formula,
    Line,
    Page,
    PageProvenance,
    Result,
    RunProvenance,
    Table,
    TableCell,
    Word,
)


def _result() -> Result:
    table = Table.from_grid([["Item", "Qty"], ["Apple", "3"]], page=1, bbox=BBox(50, 300, 250, 340))
    p1 = Page(
        number=1,
        width=612,
        height=792,
        unit="pt",
        blocks=[
            Block(TITLE, "Report", BBox(72, 50, 300, 80), 1.0, page=1, level=1, order=0, id="p1-b0", source="native"),
            Block(TEXT, "First paragraph.", BBox(72, 100, 500, 120), 0.9,
                  lines=[Line("First paragraph.", BBox(72, 100, 500, 120), 0.9, words=[Word("First", BBox(72, 100, 110, 120), 0.9), Word("paragraph.", BBox(112, 100, 200, 120), 0.9)])],
                  page=1, order=1, id="p1-b1"),
            Block(TABLE, table.text, table.bbox, 1.0, page=1, order=2, id="p1-b2", attributes={"table_index": 0}),
            Block(PAGE_FOOTER, "1", BBox(300, 760, 310, 770), 1.0, page=1, order=3, id="p1-b3"),
        ],
        tables=[table],
        provenance=PageProvenance(page=1, source="native", confidence=1.0),
    )
    p2 = Page(
        number=2,
        width=1000,
        height=1400,
        unit="px",
        blocks=[
            Block(HEADING, "Details", BBox(10, 10, 200, 40), 0.8, page=2, level=2, order=0, id="p2-b0"),
            Block(LIST_ITEM, "- point one", BBox(10, 50, 200, 70), 0.7, page=2, order=1, id="p2-b1"),
        ],
        formulas=[Formula("E = mc^2", page=2)],
        provenance=PageProvenance(page=2, source="ocr", model="ppocrv6-small", confidence=0.75,
                                  attempts=[Attempt("ppocrv6-small", confidence=0.75, accepted=True, reason="accepted")]),
    )
    return Result(pages=[p1, p2], metadata={"source_name": "doc.pdf"}, provenance=RunProvenance(textlens_version="2.0", document_hash="abc"))


def test_bbox_operations():
    a, b = BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)
    assert a.area == 100 and a.width == 10
    assert a.intersection(b) == 25
    assert a.iou(b) == pytest.approx(25 / 175)
    assert a.union(b) == BBox(0, 0, 15, 15)
    assert a.scale(2) == BBox(0, 0, 20, 20)
    assert BBox.from_points([(3, 4), (1, 9), (7, 2)]) == BBox(1, 2, 7, 9)
    assert BBox.union_all([None, a, None]) == a


def test_views_and_confidence():
    r = _result()
    assert r.page_count == 2
    assert "Report" in r.text and "Apple\t3" in r.text
    assert [b.id for b in r.blocks][:2] == ["p1-b0", "p1-b1"]
    assert len(r.tables) == 1 and len(r.formulas) == 1
    assert r.words[0].text == "Report"
    assert r.layout[0] == {"page": 1, "type": "title", "bbox": [72, 50, 300, 80], "order": 0, "id": "p1-b0", "source": "native"}
    assert 0.7 <= r.confidence <= 1.0
    assert str(r) == r.text and len(r) == 2 and r[1].number == 2


def test_round_trip_json():
    r = _result()
    again = Result.from_json(r.to_json())
    assert again.to_dict() == r.to_dict()
    assert isinstance(again.pages[0].blocks[1].bbox, BBox)
    assert again.pages[1].provenance.attempts[0].model == "ppocrv6-small"
    assert again.tables[0].grid() == [["Item", "Qty"], ["Apple", "3"]]
    assert json.loads(r.to_json())["provenance"]["schema_version"] == "2.0"


def test_markdown_export():
    md = _result().to_markdown(page_markers=True)
    assert md.startswith("<!-- page: 1 -->")
    assert "# Report" in md and "## Details" in md
    assert "| Item | Qty |" in md and "|---|---|" in md
    assert "- point one" in md
    assert "$$\nE = mc^2\n$$" in md
    assert "\n1\n" not in md  # page footer dropped by default


def test_html_export_escapes_and_annotates():
    r = _result()
    r.pages[0].blocks[1].text = "a <script> & b"
    html = r.to_html()
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert 'data-bbox="72.0,100.0,500.0,120.0"' in html
    assert "<table>" in html and "<th>Item</th>" in html


def test_table_exports_and_spans():
    t = Table(cells=[TableCell(0, 0, "H", col_span=2, is_header=True), TableCell(1, 0, "a"), TableCell(1, 1, "b")])
    assert t.n_rows == 2 and t.n_cols == 2
    assert t.grid() == [["H", "H"], ["a", "b"]]
    assert 'colspan="2"' in t.to_html()
    assert t.to_csv() == "H,H\na,b\n"
    assert t.to_records() == [{"H": "b"}] or t.to_records()[0]["H"] in ("a", "b")


@pytest.mark.parametrize("strategy", ["semantic", "block", "page"])
def test_chunking_preserves_citations(strategy):
    chunks = _result().to_chunks(max_tokens=8, overlap_tokens=0, strategy=strategy)
    assert chunks
    for c in chunks:
        assert c.text.strip()
        assert c.pages and all(p in (1, 2) for p in c.pages)
        assert c.document_id == "abc"
    assert any(b["page"] == 1 for c in chunks for b in c.bboxes)


def test_semantic_chunks_carry_section_path_and_no_orphan_overlap():
    chunks = _result().to_chunks(max_tokens=500, overlap_tokens=20, strategy="semantic")
    sections = [c.section for c in chunks]
    assert ["Report"] in sections
    assert ["Report", "Details"] in sections
    # Overlap must never create a chunk that is only carried-over text.
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))


def test_save_artifacts(tmp_path):
    written = _result().save(tmp_path, formats=("json", "md", "txt", "html"))
    for key in ("json", "md", "txt", "html", "provenance", "tables"):
        assert key in written
    assert (tmp_path / "tables" / "page-1-table-1.csv").read_text(encoding="utf-8").startswith("Item,Qty")
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["pages"][1]["model"] == "ppocrv6-small"
