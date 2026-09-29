"""Router decisions, selective OCR, validation and fallback (fake engines)."""

from __future__ import annotations

import pytest
from conftest import FAKE_CALLS, fake_system
from PIL import Image

from textlens import OCR
from textlens.core.router import PageSignals, Router
from textlens.errors import RoutingError, UnknownModelError
from textlens.runtime.system import set_system_info


def _img(w=400, h=200):
    return Image.new("RGB", (w, h), "white")


# ── router ───────────────────────────────────────────────────────────────────


def test_auto_profile_resolution(fake_models):
    set_system_info(fake_system())
    assert Router("auto").profile.name == "fast"
    set_system_info(fake_system(device_class="raspberry-pi"))
    assert Router("auto").profile.name == "edge"
    fake_models("fake-vlm", generative=True, quality=5, speed=2)
    set_system_info(fake_system(gpu_vram=8, torch_cuda=True))
    assert Router("auto").profile.name == "balanced"


def test_balanced_routes_simple_pages_fast_and_complex_pages_to_vlm(fake_models):
    fake_models("fake-vlm", generative=True, quality=5, speed=2)
    set_system_info(fake_system(gpu_vram=8, torch_cuda=True))
    router = Router("balanced")
    assert router.route(PageSignals(difficulty=0.1)).model == "fake-fast"
    complex_ = router.route(PageSignals(likely_table=True, difficulty=0.5))
    assert complex_.model == "fake-vlm" and complex_.fallbacks == ["fake-fast"]
    assert "complex page" in complex_.explain()


def test_hard_constraints_are_explained(fake_models):
    fake_models("big-vlm", generative=True, quality=5, min_vram_gb=24.0, cpu_practical=False)
    fake_models("gpu-only", generative=True, cpu=False, cpu_practical=False)
    fake_models("small-vlm", generative=True, quality=4, min_vram_gb=24.0)  # CPU-practical
    set_system_info(fake_system(gpu_vram=8, torch_cuda=True))
    router = Router("accurate")
    d = router.route()
    assert "24 GB VRAM" in d.rejected["big-vlm"]
    # Too big for the GPU but practical on CPU: allowed, and placed on CPU.
    assert "small-vlm" not in d.rejected and router.device_for(router.spec_for("small-vlm")) == "cpu"
    set_system_info(fake_system())
    d = Router("accurate").route()
    assert "requires a CUDA GPU" in d.rejected["gpu-only"]
    assert d.model == "small-vlm"  # accurate profile prefers the CPU-practical document model


def test_edge_profile_excludes_generative_models(fake_models):
    fake_models("fake-vlm", generative=True, quality=5)
    d = Router("edge").route(PageSignals(likely_table=True))
    assert d.model == "fake-fast" and "excludes generative" in d.rejected["fake-vlm"]


def test_budgets_and_license_policy(fake_models):
    fake_models("slow-model", speed=1, quality=5)
    fake_models("agpl-model", quality=5, speed=5, license_notes="x")
    import textlens.models.specs as specs_mod
    from dataclasses import replace

    specs_mod.register_spec(replace(specs_mod.get_spec("agpl-model"), license="agpl-3.0", commercial_use="restricted"), replace=True)
    d = Router("accurate", latency_budget_ms=1000).route()
    assert "exceeds" in d.rejected["slow-model"]
    d = Router("accurate", commercial_only=True).route()
    assert "agpl-model" in d.rejected and d.model != "agpl-model"
    d = Router("accurate", allowed_licenses=["apache-2.0"]).route()
    assert "agpl-model" in d.rejected


def test_remote_models_require_endpoint(fake_models, monkeypatch):
    from textlens.models.specs import ModelSpec, register_spec

    register_spec(ModelSpec(id="served-vlm", display_name="served", family="x", provider="x", description="x", backend="openai",
                            adapter="textlens.backends.openai_compat:OpenAICompatibleBackend", tasks=frozenset({"text", "markdown"}),
                            markdown=True, status="remote", quality_tier=5), replace=True)
    d = Router("server").route()
    assert "no endpoint configured" in d.rejected["served-vlm"]
    monkeypatch.setenv("TEXTLENS_OPENAI_BASE_URL", "http://localhost:9/v1")
    d = Router("server").route()
    assert d.model == "served-vlm"


def test_explicit_model_and_unknown_model(fake_models):
    d = Router(model="fake-fast").route()
    assert d.model == "fake-fast" and "explicitly requested" in d.reasons[0]
    with pytest.raises(UnknownModelError):
        Router(model="does-not-exist")
    ad_hoc = Router(model="org/any-served-model", backend="openai", endpoint="http://x/v1").route()
    assert ad_hoc.model == "org/any-served-model"


def test_no_runnable_model_raises_with_hint(fake_models):
    import textlens.models.specs as specs_mod

    specs_mod._SPECS.clear()
    with pytest.raises(RoutingError) as info:
        Router("fast").route()
    assert "textlens doctor" in (info.value.hint or "")


# ── pipeline ─────────────────────────────────────────────────────────────────


def test_selective_ocr_only_runs_on_pages_that_need_it(fake_models, fixture_pdf):
    result = OCR(profile="accurate", cache="off")(fixture_pdf)  # accurate: re-OCR scanner layers
    sources = {p.number: p.provenance.source for p in result.pages}
    assert sources == {1: "native", 2: "ocr", 3: "ocr", 4: "ocr", 5: "empty", 6: "native"}
    assert sum(FAKE_CALLS["fake-fast"]) == 3  # pages 2, 3, 4 only
    p2 = result.pages[1]
    assert p2.unit == "pt" and p2.provenance.dpi and p2.provenance.reasons == ["scanned"]
    assert p2.blocks[0].bbox.x1 <= 612  # OCR pixels mapped back to PDF points
    assert result.provenance.routing["pages_by_source"]["ocr"] == 3


def test_fast_profile_reuses_existing_ocr_layer(fake_models, fixture_pdf):
    result = OCR(profile="fast", cache="off")(fixture_pdf)
    p3 = result.pages[2]
    assert p3.provenance.source == "native" and "reused" in p3.provenance.warnings[0]
    assert sum(FAKE_CALLS["fake-fast"]) == 2


def test_ocr_modes_force_and_off(fake_models, fixture_pdf):
    forced = OCR(profile="fast", ocr="force", cache="off")(fixture_pdf)
    assert all(p.provenance.source in ("ocr", "empty") for p in forced.pages)
    FAKE_CALLS.clear()
    off = OCR(profile="fast", ocr="off", cache="off")(fixture_pdf)
    assert "fake-fast" not in FAKE_CALLS
    assert off.pages[0].provenance.source == "native"


def test_low_confidence_falls_back_to_next_model(fake_models):
    fake_models("fake-fast", confidence=0.4, default=True, speed=5)
    fake_models("fake-vlm", generative=True, quality=5, speed=2, text="Better text")
    set_system_info(fake_system(gpu_vram=8, torch_cuda=True))
    result = OCR(profile="balanced", cache="off")(_img())
    page = result.pages[0]
    assert [a.model for a in page.provenance.attempts] == ["fake-fast", "fake-vlm"]
    assert page.provenance.attempts[0].accepted is False and "confidence" in page.provenance.attempts[0].reason
    assert page.provenance.model == "fake-vlm" and page.provenance.source == "vlm"
    assert "Better text" in result.text and len(result.tables) == 1


def test_backend_error_falls_back(fake_models):
    fake_models("broken", fail=True, quality=5, speed=5)
    result = OCR(profile="fast", fallback=True, cache="off")(_img())
    attempts = result.pages[0].provenance.attempts
    assert attempts[0].model == "broken" and attempts[0].reason.startswith("error")
    assert result.pages[0].provenance.model == "fake-fast"


def test_failed_page_does_not_abort_document(fake_models, fixture_pdf):
    fake_models("fake-fast", fail=True, default=True)
    result = OCR(profile="fast", cache="off", fallback=False)(fixture_pdf)
    assert result.pages[0].provenance.source == "native"
    assert result.pages[1].provenance.needs_review and "failed" in result.pages[1].provenance.warnings[0]


def test_result_cache_hit(fake_models):
    ocr = OCR(profile="fast", cache="memory")
    ocr(_img())
    ocr(_img())
    assert sum(FAKE_CALLS["fake-fast"]) == 1
    again = ocr(_img(), cache=False)
    assert sum(FAKE_CALLS["fake-fast"]) == 2 and not again.pages[0].provenance.cached


def test_stream_pages_selection_and_read(fake_models, fixture_pdf):
    ocr = OCR(profile="fast", cache="off")
    pages = list(ocr.stream(fixture_pdf, pages=[2, 6]))
    assert [p.number for p in pages] == [2, 6]
    text = ocr.read(fixture_pdf, page=[1, 2])
    assert text.startswith("--- Page 1 ---") and "--- Page 2 ---" in text
    assert ocr.read(_img()) == "FAKE TEXT LINE"


def test_explain_dry_run_does_not_ocr(fake_models, fixture_pdf):
    decisions = OCR(profile="accurate").explain(fixture_pdf)
    assert [d.reasons[0] for d in decisions] == ["page 2: scanned", "page 3: scanned", "page 4: broken_encoding"]
    assert "fake-fast" not in FAKE_CALLS


def test_batch_directory_isolates_failures(fake_models, tmp_path):
    for name in ("a.png", "b.png"):
        _img().save(tmp_path / name)
    (tmp_path / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    results = OCR(profile="fast", cache="off").batch(tmp_path, workers=2)
    assert len(results) == 3
    assert sum(1 for r in results if isinstance(r, Exception)) == 1


def test_document_api_is_lazy(fake_models, fixture_pdf):
    import textlens

    doc = textlens.load(fixture_pdf, ocr=OCR(profile="accurate", cache="off"))
    assert doc.page_count == 6 and "fake-fast" not in FAKE_CALLS
    assert doc.page(2).provenance.source == "ocr"
    assert sum(FAKE_CALLS["fake-fast"]) == 1
    md = doc.to_markdown()
    assert sum(FAKE_CALLS["fake-fast"]) == 3  # page 2 was not OCR'd twice
    assert "# Annual Report 2026" in md
    assert doc.metadata()["title"] == "Fixture Report"
    assert len(doc.chunks(max_tokens=50)) >= 2


def test_extract_heuristic(fake_models):
    fake_models("fake-fast", text="Invoice #: INV-42\nTotal due: $1,234.50\nDate: 2026-03-04", default=True)
    ext = OCR(profile="fast", cache="off").extract(_img(), schema={"invoice_number": "string", "total": "float", "date": "date"})
    assert ext.method == "heuristic"
    assert ext.data == {"invoice_number": "INV-42", "total": 1234.5, "date": "2026-03-04"}
    assert ext.fields["total"]["page"] == 1 and ext.fields["total"]["bbox"] is not None
    assert ext.result.extraction["data"]["total"] == 1234.5
