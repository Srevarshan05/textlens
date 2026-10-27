"""ANPR: region grammars, confusion-aware correction, pipeline and tracking."""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from textlens.anpr import ANPR, REGIONS, PlateValidator, load_region, normalize
from textlens.anpr.components import Detection, Recognition, crop_plate, estimate_skew
from textlens.anpr.formats import expand_template, template_to_regex
from textlens.anpr.tracking import PlateTracker
from textlens.anpr.types import PlateRead
from textlens.core.result import BBox
from textlens.errors import ConfigurationError


def test_templates():
    assert template_to_regex("LLDDL{1,3}DDDD") == "[A-Z][A-Z][0-9][0-9][A-Z]{1,3}[0-9][0-9][0-9][0-9]"
    lengths = sorted(len(x) for x in expand_template("LLDDL{1,3}DDDD"))
    assert lengths == [9, 10, 11]


@pytest.mark.parametrize("raw,expected,fmt", [
    ("TN 01 AB 1234", "TN01AB1234", "standard"),
    ("tn-01-ab-1234", "TN01AB1234", "standard"),
    ("22 BH 1234 AA", "22BH1234AA", "bharat"),
    ("DL3CAB1234", "DL3CAB1234", "standard-short-rto"),
])
def test_india_formats(raw, expected, fmt):
    v = PlateValidator(REGIONS["in"]).validate(raw)
    assert (v.text, v.valid, v.format, v.score) == (expected, True, fmt, 1.0)


def test_position_aware_correction():
    v = PlateValidator(REGIONS["in"]).validate("TNO1AB12S4", [0.9, 0.9, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.4, 0.9])
    assert v.valid and v.text == "TN01AB1254"
    assert v.corrections == ["pos 3: O→0", "pos 9: S→5"] and v.score < 1.0
    uk = PlateValidator(REGIONS["uk"]).validate("AB1ZCDE")
    assert uk.valid and uk.text == "AB12CDE"
    assert not PlateValidator(REGIONS["in"]).validate("HELLO").valid


def test_generic_region_and_custom_region(tmp_path):
    assert PlateValidator(REGIONS["generic"]).validate("ab-123").text == "AB123"
    cfg = tmp_path / "region.json"
    cfg.write_text(json.dumps({"name": "demo", "formats": [{"name": "taxi", "regex": "^T[0-9]{4}$"}, {"name": "std", "template": "LLLDDD"}]}))
    region = load_region(str(cfg))
    val = PlateValidator(region)
    assert val.validate("T1234").format == "taxi"
    assert val.validate("ABC0I2").text == "ABC012"
    with pytest.raises(ConfigurationError):
        load_region("atlantis")
    assert normalize("Ñ-12 ö") == "N12O"


class _Det:
    name = "fake-det"

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image):
        return [Detection(BBox(*b), c) for b, c in self.boxes]


class _Rec:
    name = "fake-rec"

    def __init__(self, texts):
        self.texts = list(texts)

    def recognize(self, crops):
        out = []
        for _ in crops:
            t = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
            out.append(Recognition(t, 0.9, [0.9] * len(t)))
        return out


def test_pipeline_with_injected_stages():
    img = Image.new("RGB", (640, 480), "gray")
    anpr = ANPR(profile="edge", region="in", detector=_Det([((100, 300, 300, 350), 0.95), ((400, 50, 500, 80), 0.3)]),
                recognizer=_Rec(["TN01AB1234", "XX"]), min_confidence=0.2)
    result = anpr(img)
    assert [p.plate for p in result.plates] == ["TN01AB1234"]  # "XX" is below the length limit
    p = result.plates[0]
    assert p.valid and p.format == "standard" and p.detector == "fake-det" and p.recognizer == "fake-rec"
    assert p.confidence == pytest.approx(0.95 * 0.9, abs=1e-3)
    assert result.to_dict()["plates"][0]["bbox"] == [100, 300, 300, 350]
    annotated = result.annotate(img)
    assert annotated.size == img.size and annotated is not img


def test_require_valid_and_vehicle_assignment():
    img = Image.new("RGB", (640, 480), "gray")

    class _Vehicles:
        def detect(self, image):
            return [Detection(BBox(50, 200, 350, 450), 0.9, "car"), Detection(BBox(0, 0, 640, 480), 0.5, "bus")]

    anpr = ANPR(region="in", detector=_Det([((100, 300, 300, 350), 0.9)]), recognizer=_Rec(["HELLO1"]),
                vehicle_detector=_Vehicles(), require_valid=True, min_confidence=0.0)
    assert anpr(img).plates == []
    anpr = ANPR(region="in", detector=_Det([((100, 300, 300, 350), 0.9)]), recognizer=_Rec(["TN01AB1234"]),
                vehicle_detector=_Vehicles(), min_confidence=0.0)
    p = anpr(img).plates[0]
    assert p.vehicle_type == "car" and p.vehicle_bbox == BBox(50, 200, 350, 450)  # smallest enclosing vehicle


def test_tracking_votes_across_frames():
    tracker = PlateTracker()
    reads = ["TN01AB1234", "TN01A81234", "TN01AB1234", "7N01AB1234"]
    voted = None
    for i, text in enumerate(reads):
        box = BBox(100 + i, 300, 300 + i, 350)
        out = tracker.update(i, [PlateRead(text, 0.8, box, char_confidences=[0.8] * 10, valid=text == "TN01AB1234")])
        voted = out[0]
    assert voted.plate == "TN01AB1234" and voted.track_id == 1
    new = tracker.update(10, [PlateRead("KA05MN0042", 0.9, BBox(10, 10, 60, 30))])
    assert new[0].track_id == 2


def test_crop_and_deskew():
    from PIL import ImageDraw

    plate = Image.new("RGB", (300, 80), "white")
    ImageDraw.Draw(plate).rectangle([20, 30, 280, 50], fill="black")
    tilted = plate.rotate(8, expand=True, fillcolor=(255, 255, 255))
    assert abs(estimate_skew(tilted) + 8) <= 2 or abs(estimate_skew(tilted) - 8) <= 2
    crop = crop_plate(Image.new("RGB", (640, 480), "white"), Detection(BBox(100, 100, 300, 150), 1.0))
    assert crop.size[1] == 96


def test_unknown_profile_rejected():
    with pytest.raises(ConfigurationError):
        ANPR(profile="turbo")


@pytest.mark.skipif(os.environ.get("TEXTLENS_TEST_ANPR") != "1", reason="set TEXTLENS_TEST_ANPR=1 (downloads plate models)")
def test_real_plate_recognition_on_crop():
    pytest.importorskip("fast_plate_ocr")
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (520, 120), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 78)
    except OSError:
        font = ImageFont.load_default(size=78)
    d.text((260, 60), "KA05MN0042", fill="black", font=font, anchor="mm")
    result = ANPR(region="in", detector="full-image", min_confidence=0.0)(img)
    assert result.plates[0].plate == "KA05MN0042" and result.plates[0].valid
