"""Metrics, benchmark runner, profiler and overlays."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from textlens.evaluation.metrics import cer, exact_match, levenshtein, ned, table_cell_accuracy, text_scores, wer


def test_edit_distance_metrics():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein(["a", "b"], ["a", "c", "b"]) == 1
    assert cer("hello world", "hello world") == 0.0
    assert cer("abcd", "abce") == pytest.approx(0.25)
    assert wer("the quick brown fox", "the quick brown box") == pytest.approx(0.25)
    assert ned("abc", "") == 1.0 and ned("", "") == 0.0
    assert exact_match("Hello\n  World", "Hello World")
    assert cer("", "x") == 1.0
    s = text_scores("a b c", "a b d")
    assert s["word_edits"] == 1 and s["ref_words"] == 3


def test_table_cell_accuracy():
    ref = [["Item", "Qty"], ["Apple", "3"]]
    assert table_cell_accuracy(ref, ref) == {"cell_accuracy": 1.0, "structure_match": 1.0}
    got = table_cell_accuracy(ref, [["Item", "Qty"], ["Apple", "8"], ["x", "y"]])
    assert got["cell_accuracy"] == 0.75 and got["structure_match"] == 0.0


def _dataset(tmp_path, texts):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "ground_truth").mkdir()
    for i, t in enumerate(texts):
        Image.new("RGB", (200, 60), "white").save(tmp_path / "images" / f"s{i}.png")
        (tmp_path / "ground_truth" / f"s{i}.txt").write_text(t, encoding="utf-8")
    return tmp_path


def test_benchmark_compares_models_and_saves_measurements(fake_models, tmp_path):
    from textlens.config import get_settings
    from textlens.evaluation.benchmark import load_dataset, run_benchmark

    fake_models("fake-good", text="FAKE TEXT LINE", speed=5)
    fake_models("fake-bad", text="FAKE TXT", speed=5)
    ds = _dataset(tmp_path / "ds", ["FAKE TEXT LINE", "FAKE TEXT LINE"])
    assert len(load_dataset(ds)) == 2
    report = run_benchmark(ds, models=["fake-good", "fake-bad"], warmup=False)
    runs = {r["summary"]["label"]: r["summary"] for r in report["runs"]}
    assert runs["fake-good"]["cer"] == 0.0 and runs["fake-bad"]["cer"] > 0.3
    assert runs["fake-good"]["exact_match"] == 1.0
    assert report["recommendation"]["most_accurate"] == "fake-good"
    assert "not official" in report["note"]
    saved = json.loads((get_settings().home / "measurements.json").read_text(encoding="utf-8"))
    assert saved["fake-good"]["latency_ms_p50"] > 0


def test_benchmark_manifest_and_errors(fake_models, tmp_path):
    from textlens.errors import InputError
    from textlens.evaluation.benchmark import load_dataset

    Image.new("RGB", (100, 40), "white").save(tmp_path / "a.png")
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"file": "a.png", "text": "hi", "table": [["x"]]}) + "\n")
    samples = load_dataset(tmp_path)
    assert samples[0].text == "hi" and samples[0].table == [["x"]]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputError):
        load_dataset(empty)


def test_measurements_feed_the_router(fake_models, tmp_path):
    from textlens.config import get_settings
    from textlens.core.router import Router

    (get_settings().home).mkdir(parents=True, exist_ok=True)
    (get_settings().home / "measurements.json").write_text(json.dumps({"fake-fast": {"latency_ms_p50": 50000}}))
    from textlens.errors import RoutingError

    with pytest.raises(RoutingError) as info:
        Router("fast", latency_budget_ms=1000).route()  # measured 50 s/page exceeds the budget
    assert "50000 ms/page" in info.value.details["rejected"]["fake-fast"]


def test_profile_and_overlays(fake_models, fixture_pdf, tmp_path):
    from textlens import OCR
    from textlens.evaluation.profiler import profile_run
    from textlens.evaluation.visualize import render_overlays

    ocr = OCR(profile="accurate", cache="off")
    report = profile_run(ocr, fixture_pdf, repeat=1)
    assert report["pages"] == 6 and report["pages_by_source"]["ocr"] == 3
    assert "inspect" in report["stages_ms"] and report["cold_total_ms"] > 0
    paths = render_overlays(fixture_pdf, tmp_path / "ov", ocr=ocr, pages=[1, 2])
    assert [p.name for p in paths] == ["page-0001.png", "page-0002.png"]
    assert Image.open(paths[0]).size[0] > 100
