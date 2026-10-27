# Edge OCR

The same `OCR()` API runs on a Raspberry Pi, a Jetson, an industrial PC or a
laptop CPU. The edge path is a separate, pluggable backend
(`textlens.backends.onnx`), not special cases in the core.

## What the edge stack is

| Piece | Choice | Why |
|---|---|---|
| Engine | PP-OCRv6 Small (det + rec, ~31 MB, Apache-2.0) | accurate for printed text at a fraction of VLM cost |
| Runtime | ONNX Runtime | wheels for x86-64 and aarch64; CPU, CUDA, TensorRT, CoreML, DirectML providers |
| Pre/post-processing | numpy + Pillow | no OpenCV (avoids `opencv-python` / `-headless` conflicts and large wheels) |
| PDFs | PDFium via pypdfium2 | native text without OCR; rendering only for pages that need it |

Install footprint: the core wheel plus numpy, Pillow, pypdfium2 and
onnxruntime. No PyTorch.

## Setup

```bash
pip install textlens-ocr
textlens setup --profile edge --yes      # downloads + verifies the model, saves the profile
textlens ocr sample.jpg
```

Offline devices: install once with network, or copy `~/.cache/textlens/models`
from another machine, then set `TEXTLENS_OFFLINE=1`.

## The `edge` profile

- ONNX PP-OCR only (generative models excluded)
- detector input capped at 960 px, recognition batches of 4
- PDFs rendered at 150 dpi
- scanner OCR layers reused instead of re-OCR'd

Tune per device:

```python
OCR(profile="edge", threads=4)               # ORT threads (default: physical cores, max 8)
OCR(profile="edge", det_limit=736)           # faster, less accurate on small text
OCR(profile="edge", low_memory=True)         # ≤1 GB boards: no ORT arena, ~3× slower detection
OCR(profile="edge", device="tensorrt")       # Jetson with TensorRT-enabled ORT
```

Measure on the device:

```bash
textlens profile sample.pdf --repeat 3
```

## Device guides

- [Raspberry Pi](raspberry-pi.md)
- [Jetson](jetson.md)

## Beyond text

`textlens.anpr` ([number plates](../tasks/anpr.md)) runs on the same ONNX
Runtime and suits edge cameras.
