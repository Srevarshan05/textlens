# GPU setup (NVIDIA)

TextLens uses the GPU in two independent ways:

| Engine | Runtime | Install |
|---|---|---|
| PP-OCRv6 (default) | ONNX Runtime CUDA provider | `pip uninstall -y onnxruntime && pip install onnxruntime-gpu` |
| Document VLMs (GLM-OCR, LightOnOCR, HunyuanOCR, SmolVLM) | PyTorch + Transformers | CUDA torch wheel, then `pip install "textlens-ocr[gpu]"` |

## 1. Check the driver

```bash
nvidia-smi          # shows the maximum CUDA version the driver supports
textlens doctor     # prints matching install commands
```

## 2. PyTorch with CUDA

PyPI's default torch is CPU-only on Windows. Pick the index matching your driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # driver ≥ CUDA 12.4
pip install "textlens-ocr[gpu]"
```

GLM-OCR, LightOnOCR and HunyuanOCR need `transformers>=5`.

## 3. Verify

```bash
textlens doctor --deep     # imports torch, confirms torch.cuda.is_available()
textlens models install glm-ocr
textlens ocr hard-table.png --model glm-ocr --explain
```

## Choosing a model by VRAM

| VRAM | Sensible setup |
|---|---|
| none / CPU | `fast` profile (PP-OCR); SmolVLM for rough VLM output |
| 4 GB | PP-OCR on GPU; GLM-OCR works but tight |
| 6–8 GB | `balanced`: PP-OCR + GLM-OCR for complex pages |
| 8–16 GB | + LightOnOCR, HunyuanOCR |
| ≥ 16 GB / server | serve DeepSeek-OCR-2 or PaddleOCR-VL with vLLM (`server` profile) |

These are starting points derived from model sizes, not measurements —
confirm with `textlens profile` and `textlens benchmark` on your pages.

## Memory

- One copy of each model per process, shared by all threads and requests.
- `OCR(memory_budget_gb=4)` keeps the router away from larger models.
- Models too large for the GPU but practical on CPU run on CPU instead of
  failing with out-of-memory.
- `OCR().unload()` frees loaded models.
- Several GPUs: `OCR(device="cuda:1")`, or one server replica per GPU.

## TensorRT

ONNX engines can use the TensorRT execution provider: install an
`onnxruntime-gpu` build with TensorRT and set `TEXTLENS_ORT_TENSORRT=1` (or
`device="tensorrt"`). Engines are cached under `$TEXTLENS_HOME/trt-cache`;
the first run builds them (slow), later runs load them.
