# NVIDIA Jetson

Jetson Nano (original), Xavier and Orin boards run TextLens's edge stack. GPU
acceleration needs NVIDIA's ONNX Runtime build for your JetPack — PyPI's
`onnxruntime` is CPU-only on aarch64.

## JetPack 6 (Orin)

```bash
python3 -m venv ~/textlens && source ~/textlens/bin/activate
pip install textlens-ocr
# GPU ONNX Runtime from the Jetson AI Lab index (match JetPack/CUDA):
pip uninstall -y onnxruntime
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
textlens doctor      # "Backends: ONNX Runtime … [TensorRT, CUDA, CPU]"
textlens setup --profile edge --yes
```

Then:

```python
from textlens import OCR
ocr = OCR(profile="edge", device="cuda")        # CUDA provider
ocr = OCR(profile="edge", device="tensorrt")    # TensorRT provider (engine cache in $TEXTLENS_HOME/trt-cache)
```

The first TensorRT run builds engines for the model (can take minutes);
subsequent runs load the cached engines. Input shapes vary with page size,
so keep `det_limit` fixed for best engine reuse.

## Original Jetson Nano (JetPack 4, Python 3.6)

JetPack 4 ships Python 3.6; TextLens requires Python ≥ 3.10. Run TextLens in
a container with a newer Python (see below) or use CPU inference from a
Python 3.10+ environment.

## Docker

```bash
docker build -f deploy/docker/Dockerfile.jetson -t textlens:jetson .
docker run --runtime nvidia -p 8000:8000 -v textlens-models:/models textlens:jetson
```

The Jetson Dockerfile is a reference: adjust `BASE` and `JETSON_PIP_INDEX`
to your JetPack version and build on the device. It is not CI-tested.

## Document VLMs on Orin

Orin boards with 32–64 GB unified memory can run the `gpu` extra with
NVIDIA's PyTorch wheels for JetPack. Start with PP-OCR (`edge`/`fast`) and
add a VLM only for pages that need it (`balanced`), measuring with
`textlens profile`.
