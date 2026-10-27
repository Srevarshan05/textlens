"""The 2.0 CLI: commands, output formats and exit codes."""

from __future__ import annotations

import json

from PIL import Image

from textlens.cli import main
from textlens.cli.output import parse_pages


def run(capsys, *argv):
    code = 0
    try:
        main(list(argv))
    except SystemExit as exc:
        code = exc.code or 0
    out = capsys.readouterr()
    return code, out.out, out.err


def test_parse_pages():
    assert parse_pages("1-3,5,5") == [1, 2, 3, 5]
    assert parse_pages("") is None
    assert parse_pages("2-")[:3] == [2, 3, 4]


def test_help_and_version(capsys):
    code, out, _ = run(capsys, "--help")
    assert code == 0 and "textlens ocr" in out and "inspect" in out
    code, out, _ = run(capsys, "--version")
    assert code == 0 and out.startswith("textlens 2.")


def test_models_list_and_info_json(capsys):
    code, out, _ = run(capsys, "models", "list", "--json")
    ids = {m["id"] for m in json.loads(out)}
    assert code == 0 and {"ppocrv6-small", "glm-ocr", "paddleocr-vl"} <= ids
    code, out, _ = run(capsys, "models", "info", "glm-ocr", "--json")
    info = json.loads(out)
    assert info["license"] == "mit" and "table" in info["tasks"]
    code, out, _ = run(capsys, "models", "search", "table", "--json")
    assert any(m["id"] == "glm-ocr" for m in json.loads(out))


def test_unknown_model_exit_code(capsys):
    code, _, err = run(capsys, "models", "info", "nope-model")
    assert code == 1 and "Unknown model" in err


def test_missing_file_exit_code(capsys, fake_models):
    code, _, err = run(capsys, "ocr", "does-not-exist.png")
    assert code == 2 and "not found" in err


def test_ocr_formats_and_output_file(capsys, fake_models, tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (300, 100), "white").save(img)
    code, out, _ = run(capsys, "-q", "ocr", str(img))
    assert code == 0 and out.strip() == "FAKE TEXT LINE"
    code, out, _ = run(capsys, "ocr", str(img), "-f", "json")
    assert json.loads(out)["pages"][0]["provenance"]["model"] == "fake-fast"
    dest = tmp_path / "out.md"
    code, _, _ = run(capsys, "ocr", str(img), "-f", "markdown", "-o", str(dest))
    assert code == 0 and "FAKE TEXT LINE" in dest.read_text(encoding="utf-8")
    code, _, err = run(capsys, "ocr", str(img), "--explain")
    assert "Selected: fake-fast" in err


def test_inspect_json(capsys, fixture_pdf, tmp_path):
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(fixture_pdf)
    code, out, _ = run(capsys, "inspect", str(pdf), "--json")
    data = json.loads(out)
    assert code == 0 and data["pdf_type"] == "mixed" and data["pages_needing_ocr"] == [2, 3, 4]


def test_extract_command(capsys, fake_models, tmp_path):
    fake_models("fake-fast", text="Total: $12.50", default=True)
    img = tmp_path / "r.png"
    Image.new("RGB", (300, 100), "white").save(img)
    code, out, _ = run(capsys, "extract", str(img), "-s", "total")
    assert code == 0 and json.loads(out) == {"total": 12.5}


def test_doctor_json(capsys):
    code, out, _ = run(capsys, "doctor", "--json")
    data = json.loads(out)
    assert code == 0 and "hardware" in data and data["models"]
    assert {m["id"] for m in data["models"]} >= {"ppocrv6-small", "glm-ocr"}


def test_setup_non_interactive_writes_profile(capsys, fake_models):
    code, _, err = run(capsys, "setup", "--profile", "edge", "--yes", "--no-download")
    assert code == 0 and "edge" in err
    from textlens.config import get_settings

    assert get_settings().profile == "edge"


def test_batch_no_dashboard_with_resume(capsys, fake_models, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for n in ("a.png", "b.png"):
        Image.new("RGB", (200, 80), "white").save(src / n)
    out_dir = tmp_path / "out"
    code, _, _ = run(capsys, "batch", str(src), "--no-dashboard", "-o", str(out_dir), "-w", "2")
    assert code == 0
    data = json.loads((out_dir / "a_ocr.json").read_text(encoding="utf-8"))
    assert data["text"] == "FAKE TEXT LINE" and data["result"]["pages"][0]["provenance"]["model"] == "fake-fast"
    code, _, err = run(capsys, "batch", str(src), "--no-dashboard", "-o", str(out_dir), "--resume")
    assert code == 0 and "2 skipped" in err
