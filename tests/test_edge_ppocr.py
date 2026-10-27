"""Edge OCR subsystem: DBNet post-processing (no weights) and PP-OCRv6 (real weights)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURES

from textlens.backends.onnx.dbnet import boxes_from_probability, convex_hull, min_area_rect, order_quad


def test_dbnet_finds_rectangles_in_probability_map():
    prob = np.zeros((100, 200), np.float32)
    prob[20:30, 20:120] = 0.9  # a text line
    prob[60:70, 50:90] = 0.8  # another
    prob[90:92, 5:8] = 0.95  # too small → ignored
    boxes = boxes_from_probability(prob, (2.0, 2.0), unclip_ratio=1.5)
    assert len(boxes) == 2
    quads = sorted((q for q, _ in boxes), key=lambda q: q[0][1])
    x0, y0 = quads[0].min(0)
    x1, y1 = quads[0].max(0)
    assert x0 < 40 < 240 < x1 and y0 < 40 < 60 < y1  # scaled ×2 and unclipped


def test_rotated_component_gets_rotated_rectangle():
    pts = np.array([[0, 0], [10, 10], [20, 0], [10, -10]], np.float64)
    cx, cy, w, h, angle = min_area_rect(convex_hull(pts))
    assert w * h == pytest.approx(200, rel=0.01)
    ordered = order_quad(np.array([[5, 5], [0, 0], [5, 0], [0, 5]], np.float64))
    assert ordered.tolist() == [[0, 0], [5, 0], [5, 5], [0, 5]]


def test_provider_selection_cpu():
    pytest.importorskip("onnxruntime")
    from textlens.backends.onnx.runtime import select_providers

    assert select_providers("cpu") == ["CPUExecutionProvider"]
    assert select_providers(None)[-1] == "CPUExecutionProvider"


@pytest.mark.models
def test_ppocr_reads_real_images():

    from textlens import OCR

    result = OCR(model="ppocrv6-small", cache="off")(FIXTURES / "test-image-ocr.png")
    assert result.text.splitlines() == ["Hello World.", "Using Tesseract's OCR.", "From srcmake."]
    page = result.pages[0]
    assert page.provenance.model == "ppocrv6-small" and page.provenance.backend == "onnxruntime"
    assert page.confidence > 0.9
    line = page.blocks[0].lines[0]
    assert line.polygon and len(line.words) == 2 and line.words[1].text == "World."


@pytest.mark.models
def test_ppocr_invoice_row_major_and_edge_profile():
    from textlens import OCR

    result = OCR(profile="edge", cache="off")(FIXTURES / "invoice.png")
    text = result.text
    assert "INVOICE" in text and "TOTAL DUE $5.00" in text
    # Tabular layout stays row-major: label and amounts on one line.
    assert any("Design Service" in ln and "$0.00" in ln for ln in text.splitlines())


@pytest.mark.models
def test_scanned_pdf_end_to_end():
    import pdf_factory as F

    from textlens import OCR

    pdf = F.build_pdf([F.scanned_page(["Scanned invoice 4471", "Total due: 5,240.00 EUR"])])
    result = OCR(profile="fast", cache="off")(pdf)
    assert "Scanned invoice 4471" in result.text and "5,240.00" in result.text
    b = result.pages[0].blocks[0]
    assert result.pages[0].unit == "pt" and 0 <= b.bbox.x0 < b.bbox.x1 <= 612
