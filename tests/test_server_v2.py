"""REST API: auth, limits, jobs, streaming, backpressure, legacy + OpenAI facades."""

from __future__ import annotations

import base64
import io
import json
import threading
import time

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from textlens.serving.app import create_app  # noqa: E402
from textlens.serving.settings import ServerSettings  # noqa: E402

KEY = {"X-API-Key": "k1"}


def _png(w=200, h=80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _client(**kw) -> TestClient:
    kw.setdefault("api_keys", ("k1",))
    return TestClient(create_app(ServerSettings(**kw)))


def test_health_is_public_and_ocr_requires_key(fake_models):
    with _client() as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.post("/ocr", files={"file": ("a.png", _png())}).status_code == 401
        assert c.post("/ocr", files={"file": ("a.png", _png())}, headers={"X-API-Key": "wrong"}).status_code == 401
        ok = c.post("/ocr", files={"file": ("a.png", _png())}, headers={"Authorization": "Bearer k1"})
        assert ok.status_code == 200 and ok.json()["text"] == "FAKE TEXT LINE"
        assert ok.headers["x-content-type-options"] == "nosniff" and ok.headers["x-request-id"]


def test_custom_auth_hook(fake_models):
    app = create_app(ServerSettings(), auth=lambda req: "svc" if req.headers.get("x-gateway-user") else None)
    with TestClient(app) as c:
        assert c.post("/ocr", files={"file": ("a.png", _png())}).status_code == 401
        assert c.post("/ocr", files={"file": ("a.png", _png())}, headers={"x-gateway-user": "alice"}).status_code == 200


def test_formats_and_validation(fake_models):
    with _client(max_upload_mb=1) as c:
        md = c.post("/ocr", files={"file": ("a.png", _png())}, data={"format": "markdown"}, headers=KEY)
        assert md.headers["content-type"].startswith("text/markdown")
        assert c.post("/ocr", files={"file": ("a.png", _png())}, data={"format": "text"}, headers=KEY).text == "FAKE TEXT LINE"
        bad = c.post("/ocr", files={"file": ("a.png", b"GIF-not-really")}, headers=KEY)
        assert bad.status_code == 415 and bad.json()["error"] == "unsupported_input"
        big = c.post("/ocr", files={"file": ("a.png", b"\x89PNG" + b"0" * 1_200_000)}, headers=KEY)
        assert big.status_code == 413
        assert c.post("/ocr", headers=KEY).status_code == 400
        assert c.post("/ocr", data={"url": "https://example.com/a.png"}, headers=KEY).json()["error"] == "input_error"


def test_legacy_routes_block_local_files(fake_models, tmp_path):
    with _client() as c:
        r = c.post("/api/v1/ocr", data={"image_url": str(tmp_path / "x.png")}, headers=KEY)
        assert r.status_code == 400 and "Local file paths" in r.json()["message"]
        ok = c.post("/api/v1/ocr", files={"file": ("a.png", _png())}, headers=KEY).json()
        assert ok["status"] == "success" and ok["is_pdf"] is False and ok["text"] == "FAKE TEXT LINE"
    img = tmp_path / "ok.png"
    img.write_bytes(_png())
    with _client(allowed_root=str(tmp_path)) as c:
        assert c.post("/api/v1/ocr/json-payload", json={"image_url": "ok.png"}, headers=KEY).json()["status"] == "success"


def test_async_jobs_and_result(fake_models, fixture_pdf):
    with _client() as c:
        r = c.post("/ocr", files={"file": ("d.pdf", fixture_pdf)}, data={"async": "true"}, headers=KEY)
        assert r.status_code == 202
        jid = r.json()["job_id"]
        for _ in range(100):
            st = c.get(f"/jobs/{jid}", headers=KEY).json()
            if st["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert st["status"] == "succeeded" and st["progress"]["pages_done"] == 6
        res = c.get(f"/jobs/{jid}/result", headers=KEY).json()
        assert len(res["pages"]) == 6 and res["pages"][0]["provenance"]["source"] == "native"
        assert c.get("/jobs/nope", headers=KEY).status_code == 404


def test_streaming_ndjson(fake_models, fixture_pdf):
    with _client() as c:
        r = c.post("/ocr", files={"file": ("d.pdf", fixture_pdf)}, data={"stream": "true", "pages": "1-2"}, headers=KEY)
        events = [json.loads(line) for line in r.text.strip().splitlines()]
        assert [e["type"] for e in events] == ["page", "page", "done"]
        assert events[1]["page"]["number"] == 2


def test_backpressure_returns_503(fake_models):
    gate = threading.Event()
    with _client(workers=1, max_queue=0) as c:
        svc = c.app.state.textlens
        svc.jobs.submit(lambda job: gate.wait(5))  # occupy the only slot
        r = c.post("/ocr", files={"file": ("a.png", _png())}, headers=KEY)
        assert r.status_code == 503 and r.headers["retry-after"]
        gate.set()


def test_rate_limit(fake_models):
    with _client(rate_limit=0.001, rate_burst=1) as c:
        assert c.get("/models", headers=KEY).status_code == 200
        r = c.get("/models", headers=KEY)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_batch_extract_inspect_metrics(fake_models, fixture_pdf):
    with _client() as c:
        r = c.post("/batch", files=[("files", ("a.png", _png())), ("files", ("b.pdf", fixture_pdf))], headers=KEY)
        jid = r.json()["job_id"]
        for _ in range(100):
            if c.get(f"/jobs/{jid}", headers=KEY).json()["status"] == "succeeded":
                break
            time.sleep(0.05)
        items = c.get(f"/jobs/{jid}/result", headers=KEY).json()["items"]
        assert [i["status"] for i in items] == ["succeeded", "succeeded"]
        insp = c.post("/inspect", files={"file": ("d.pdf", fixture_pdf)}, headers=KEY).json()
        assert insp["pdf_type"] == "mixed" and insp["pages_needing_ocr"] == [2, 3, 4]
        ext = c.post("/extract", files={"file": ("a.png", _png())}, data={"schema": "total,date"}, headers=KEY)
        assert ext.status_code == 200 and set(ext.json()["data"]) == {"total", "date"}
        metrics = c.get("/metrics").text
        assert "textlens_pages_total" in metrics and "textlens_jobs_total" in metrics


def test_openai_facade(fake_models):
    b64 = base64.b64encode(_png()).decode()
    body = {"model": "textlens", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    with _client() as c:
        r = c.post("/v1/chat/completions", json=body, headers=KEY)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "FAKE TEXT LINE"
        assert "textlens" in [m["id"] for m in c.get("/v1/models", headers=KEY).json()["data"]]
        bad = dict(body, messages=[{"role": "user", "content": "no image"}])
        assert c.post("/v1/chat/completions", json=bad, headers=KEY).status_code == 400


def test_legacy_create_app_signature():
    from textlens.server import create_app as legacy_create_app

    with TestClient(legacy_create_app()) as c:
        root = c.get("/").json()
        assert root["name"] == "TextLens OCR REST Service" and root["status"] == "online"
