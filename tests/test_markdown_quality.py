"""VLM Markdown parsing and text-quality (garbled layer) detection."""

from __future__ import annotations

from textlens.core import quality
from textlens.core.markdown import parse_html_table, parse_markdown

VLM_OUTPUT = """# Invoice 4471

## Line items

| Item | Qty | Price |
|------|----:|------:|
| Apple | 3 | 1.20 |
| Pipe \\| fitting | 1 | 9.99 |

- first bullet
1. numbered

$$
\\int_0^1 x^2 dx = \\frac{1}{3}
$$

<table><tr><th rowspan="2">Region</th><th colspan="2">Sales</th></tr><tr><td>Q1</td><td>Q2</td></tr><tr><td>EU</td><td>10</td><td>12</td></tr></table>

```
code block
```

Plain paragraph that
spans two lines.
"""


def test_parse_markdown_structure():
    blocks, tables, formulas = parse_markdown(VLM_OUTPUT, page=3, model="glm-ocr", backend="transformers", confidence=None)
    types = [b.type for b in blocks]
    assert types[:2] == ["title", "heading"]
    assert "list_item" in types and "code" in types and types[-1] == "text"
    assert len(tables) == 2 and len(formulas) == 1
    assert tables[0].grid()[2] == ["Pipe | fitting", "1", "9.99"]
    assert all(b.page == 3 and b.source == "vlm" and b.model == "glm-ocr" for b in blocks)
    assert "\\frac{1}{3}" in formulas[0].latex
    assert blocks[-1].text == "Plain paragraph that\nspans two lines."
    assert [b.order for b in blocks] == list(range(len(blocks)))


def test_html_table_spans():
    t = parse_html_table('<table><tr><th rowspan="2">Region</th><th colspan="2">Sales</th></tr><tr><td>Q1</td><td>Q2</td></tr></table>')
    assert t is not None
    assert t.grid() == [["Region", "Sales", "Sales"], ["Region", "Q1", "Q2"]]


def test_malformed_markdown_never_raises():
    for text in ("", "|", "| a |\n|---", "$$ unclosed", "<table><tr><td>x", "```\nno close"):
        parse_markdown(text)


CLEAN = (
    "The quarterly report describes revenue growth across all regions. Operating margins improved "
    "because logistics costs declined, while investment in research continued at a steady pace. "
    "Management expects similar results next year, subject to market conditions and currency movements."
)


def test_clean_text_is_not_garbled():
    report = quality.assess_text(CLEAN * 2)
    assert report.ok and not report.signals


def test_cipher_garbled_text_detected():
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    shifted = "".join(alphabet[(alphabet.index(c) * 7 + 3) % 52] if c.isalpha() else c for c in CLEAN * 2)
    assert quality.looks_cipher_garbled(shifted)
    assert quality.assess_text(shifted).garbled


def test_replacement_character_density():
    assert quality.replacement_needs_ocr("�� bad")
    assert not quality.replacement_needs_ocr(CLEAN + " � ")  # one math glyph in prose is fine


def test_symbol_soup_and_leaders():
    assert quality.is_symbol_soup("----1-.-.-.___  --.-. .._ I_---." * 4 + "%%%$$$@@@" * 6)
    assert not quality.is_symbol_soup("Chapter 1 .................... 5\nChapter 2 .................... 9" * 3)


def test_dollar_as_space_and_private_use():
    assert quality.has_dollar_as_space("The$quick$brown$fox$jumps$over$the$lazy$dog$again$and$again$today$now")
    assert not quality.has_dollar_as_space("Total $5.00, tax $0.40, tip $1.00")
    assert quality.has_private_use_run("abc  def")


def test_cid_garbage_c1_controls():
    assert quality.is_cid_garbage("ab\x85\x86cd\x87\x88efgh")
