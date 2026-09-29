"""Input normalisation, sniffing, limits and configuration precedence."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from textlens.config import Settings, get_settings, reload_settings, write_config_file
from textlens.errors import InputError, InputNotFoundError, InputTooLargeError, UnsupportedInputError
from textlens.inputs.source import open_source, sniff


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_sniff_by_magic_bytes_not_extension(tmp_path):
    p = tmp_path / "misnamed.pdf"
    p.write_bytes(_png_bytes())
    src = open_source(p)
    assert src.kind == "image" and src.mime == "image/png"
    assert sniff(b"%PDF-1.7\n...")[0] == "pdf"
    with pytest.raises(UnsupportedInputError):
        sniff(b"hello world")


def test_all_input_types(tmp_path, fixture_pdf):
    img = Image.new("RGB", (8, 8), "white")
    assert open_source(img).kind == "image"
    assert open_source(np.zeros((8, 8, 3), np.uint8)).kind == "image"
    assert open_source(np.zeros((8, 8), np.float32)).image.mode == "RGB"
    assert open_source(fixture_pdf).kind == "pdf"
    assert open_source(io.BytesIO(fixture_pdf)).kind == "pdf"
    p = tmp_path / "doc.pdf"
    p.write_bytes(fixture_pdf)
    src = open_source(str(p))
    assert src.path == p and src.data is None and src.size == len(fixture_pdf)
    assert len(src.sha256) == 64 and src.sha256 == open_source(fixture_pdf).sha256


def test_docx_detected_from_zip_content(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<xml/>")
    assert open_source(buf.getvalue(), name="upload.bin").kind == "docx"


def test_errors_are_structured_and_backwards_compatible(tmp_path):
    with pytest.raises(FileNotFoundError):  # InputNotFoundError is a FileNotFoundError
        open_source(tmp_path / "missing.png")
    with pytest.raises(InputNotFoundError) as info:
        open_source(tmp_path / "missing.png")
    assert info.value.code == "input_not_found" and info.value.http_status == 404
    with pytest.raises(ValueError):  # UnsupportedInputError is a ValueError
        open_source(12345)
    with pytest.raises(InputTooLargeError):
        open_source(b"%PDF-" + b"0" * 2000, max_bytes=100)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(InputError):
        open_source(empty)


def test_allowed_root_blocks_path_traversal(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / "ok.png").write_bytes(_png_bytes())
    outside = tmp_path / "secret.png"
    outside.write_bytes(_png_bytes())
    assert open_source("ok.png", allowed_root=inside).name == "ok.png"
    with pytest.raises(InputError):
        open_source("../secret.png", allowed_root=inside)
    with pytest.raises(InputError):
        open_source(str(outside), allowed_root=inside)


def test_urls_can_be_disabled():
    with pytest.raises(InputError):
        open_source("https://example.com/x.png", allow_urls=False)


def test_url_guard_rejects_private_addresses():
    from textlens.inputs.source import default_url_guard

    with pytest.raises(InputError):
        default_url_guard("http://127.0.0.1/secret")
    with pytest.raises(InputError):
        default_url_guard("http://localhost:8080/")


def test_settings_precedence(tmp_path):
    env = {"TEXTLENS_HOME": str(tmp_path), "TEXTLENS_PROFILE": "edge", "TEXTLENS_MAX_PAGES": "12"}
    s = Settings.from_env(env, file_values={"profile": "accurate", "device": "cpu"})
    assert s.profile == "edge"  # env beats file
    assert s.device == "cpu"  # file beats default
    assert s.max_pages == 12
    assert s.models_dir == tmp_path / "models"
    bad = Settings.from_env({"TEXTLENS_HOME": str(tmp_path), "TEXTLENS_PROFILE": "nope", "TEXTLENS_MAX_PAGES": "x"}, file_values={})
    assert bad.profile == "auto" and bad.max_pages == 5000
    np_ = Settings.from_env({"TEXTLENS_HOME": str(tmp_path), "TEXTLENS_NO_PERSIST": "1", "TEXTLENS_RESULT_CACHE": "disk"}, file_values={})
    assert np_.result_cache == "memory"  # no-persist forbids disk caching
    assert Settings.from_env({"HF_TOKEN": "t", "TEXTLENS_HOME": str(tmp_path)}, file_values={}).to_dict()["hf_token"] == "***"


def test_config_file_round_trip():
    path = write_config_file({"profile": "fast"})
    assert json.loads(path.read_text(encoding="utf-8"))["profile"] == "fast"
    assert get_settings().profile == "fast"
    write_config_file({"profile": "auto"})
    reload_settings()


def test_import_has_no_side_effects(tmp_path):
    import subprocess
    import sys

    code = (
        "import os, sys; os.environ['TEXTLENS_HOME']=sys.argv[1]; import textlens; "
        "from textlens import OCR, Result; import textlens.models.manager; "
        "assert not os.path.exists(os.path.join(sys.argv[1], 'models')), 'import created dirs'; "
        "assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules and 'fastapi' not in sys.modules"
    )
    home = tmp_path / "home"
    r = subprocess.run([sys.executable, "-c", code, str(home)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
