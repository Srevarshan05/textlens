# Number-plate recognition (ANPR)

ANPR in TextLens is a dedicated pipeline, not "run OCR on a car photo":

```text
image ─► vehicle detection (optional) ─► plate detection ─► crop + deskew / perspective
      ─► plate recognition ─► region-format validation + correction ─► combined confidence
      ─► video: tracking + per-character temporal voting
```

It lives in `textlens.anpr` and is independent of the document pipeline.

```bash
pip install "textlens-ocr[anpr]"
```

```python
from textlens import ANPR

anpr = ANPR(profile="edge", region="in")
result = anpr("vehicle.jpg")
for p in result.plates:
    print(p.plate, p.confidence, p.valid, p.format, p.bbox)
```

```json
{"plate": "TN01AB1234", "confidence": 0.93, "valid": true, "format": "standard",
 "bbox": [448, 468, 833, 566], "detection_confidence": 0.97, "recognition_confidence": 0.99,
 "char_confidences": [...], "corrections": [], "vehicle_type": "car", "track_id": null}
```

```bash
textlens anpr car.jpg --region in --annotate out/
```

## Models

| Stage | Default (anpr extra) | Without the extra |
|---|---|---|
| Plate detection | open-image-models YOLOv9 (ONNX, MIT): `t-384` edge · `t-640` balanced · `s-608` accurate | PP-OCR text regions filtered by plate geometry |
| Recognition | fast-plate-ocr CCT models trained on plates (ONNX, MIT), per-character confidence, region head | PP-OCRv6 recognition |
| Vehicle type | RF-DETR COCO (`vehicle_detector=True`) | — |

Models download on first use (a few MB each). All run on CPU through ONNX
Runtime; with `onnxruntime-gpu` they use CUDA.

## Regions: validation and correction

Plate formats are configuration, not code. Built-in regions: `generic`, `in`
(standard, short RTO, legacy, BH series), `uk`, `de`, `fr`, `it`, `es`, `nl`,
`eu`, `us`, `br`, `ae`.

Knowing whether a position expects a letter or a digit fixes typical OCR
confusions: `TNO1AB12S4` → `TN01AB1254` (`O→0`, `S→5`), preferring to change
low-confidence characters. Corrections are listed in `plate.corrections`;
`plate.valid` says whether the result matches a format.

Custom regions — JSON (or YAML with PyYAML):

```json
{
  "name": "my-city",
  "min_length": 5,
  "formats": [
    {"name": "private", "template": "LLLDDD"},
    {"name": "taxi", "regex": "^T[0-9]{4}$"}
  ]
}
```

Template classes: `L` letter, `D` digit, `A` either, `{m,n}` repetition.

```python
ANPR(region="my-city.json", require_valid=True)
```

## Video

```python
for result in ANPR(profile="edge", region="uk").stream(frames):
    for p in result.plates:
        print(result.frame, p.track_id, p.plate, p.confidence)
```

Plates are tracked by box overlap; each track's reading is a
confidence-weighted per-character vote over recent frames, so one blurred
frame does not flip the plate.

## Custom stages

Every stage is injectable:

```python
ANPR(detector=my_detector, recognizer=my_recognizer)   # objects with detect() / recognize()
ANPR(detector="full-image")                            # inputs are already plate crops
```

## Accuracy

Practical accuracy depends on camera placement, resolution, motion blur,
illumination and the plate styles in your region. Validate on footage from
your cameras before deployment: collect crops with ground-truth plates and
use `textlens benchmark` (manifest with `text` = plate) or the ANPR API in a
loop. TextLens does not publish plate accuracy figures it has not measured.
