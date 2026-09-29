# Installation

```bash
pip install textlens-ocr
```

The import name is `textlens`. (The distribution is published as
`textlens-ocr` because the shorter name collides with an existing PyPI
project.)

That is all you need to OCR images and PDFs on any CPU. The core install is
small and CUDA-free:

| Package | Why |
|---|---|
| `pypdfium2` | PDF inspection, native text extraction, rendering |
| `onnxruntime` | runs the default OCR engine (PP-OCRv6) on CPU |
| `numpy`, `pillow` | image handling |

No PyTorch, no CUDA toolkit, no OpenCV. The first OCR call downloads the
default model (~31 MB) and verifies its SHA-256 checksum. Native-text PDFs
never trigger a download.

Requires Python 3.10+.

## Optional features

Install only what you use:

| Extra | Adds | Install |
|---|---|---|
| `gpu` | local document VLMs (GLM-OCR, LightOnOCR, HunyuanOCR, SmolVLM) via PyTorch/Transformers | `pip install "textlens-ocr[gpu]"` |
| `server` | REST API (`textlens serve`) | `pip install "textlens-ocr[server]"` |
| `documents` | DOCX and PPTX input | `pip install "textlens-ocr[documents]"` |
| `anpr` | number-plate detection/recognition models | `pip install "textlens-ocr[anpr]"` |
| `mcp` | MCP server for AI agents | `pip install "textlens-ocr[mcp]"` |
| `ui` | rich terminal output, system telemetry | `pip install "textlens-ocr[ui]"` |
| `vllm` | vLLM, to self-host large models (Linux) | `pip install "textlens-ocr[vllm]"` |
| `all` | everything above except `vllm` | `pip install "textlens-ocr[all]"` |

Combine extras: `pip install "textlens-ocr[server,documents]"`.

TextLens never runs `pip install` behind your back. If a feature needs a
missing package, the error tells you the exact command.

## GPU (NVIDIA)

PyTorch wheels on PyPI are CPU-only on Windows. Install a CUDA build first,
then the `gpu` extra:

```bash
textlens doctor            # prints the right torch command for your driver
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "textlens-ocr[gpu]"
```

See [GPU setup](../deployment/gpu.md).

## Models

Model weights are never part of the Python package:

```bash
textlens models list                 # catalog with install status
textlens models install glm-ocr      # download (and verify) weights
textlens models info glm-ocr         # capabilities, VRAM, license, limitations
textlens models remove glm-ocr
```

Weights live in `~/.cache/textlens/models` (override with
`TEXTLENS_MODELS_DIR`). For air-gapped machines, copy that folder and set
`TEXTLENS_OFFLINE=1`.

## Check the installation

```bash
textlens doctor
textlens setup      # optional guided configuration
```

## Upgrading from 0.x

See the [migration guide](../migration.md). Installed models stay installed.
