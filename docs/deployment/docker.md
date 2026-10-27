# Docker

Docker is optional; Python users never need it. Images live in
`deploy/docker/`.

| Image | Dockerfile | Contents |
|---|---|---|
| CPU / server | `Dockerfile` | slim Python, server + documents extras, PP-OCRv6 baked in, no PyTorch |
| GPU (CUDA) | `Dockerfile.gpu` | PyTorch CUDA base, `gpu` extra, `onnxruntime-gpu` |
| Jetson | `Dockerfile.jetson` | L4T JetPack base, Jetson ONNX Runtime (reference, not CI-tested) |

> The Dockerfiles are exercised by CI on Linux (build + health check for the
> CPU image). The GPU and Jetson images are provided as references; build and
> test them on your target hardware.

## CPU

```bash
docker build -f deploy/docker/Dockerfile -t textlens .
docker run -p 8000:8000 -e TEXTLENS_API_KEYS=change-me textlens
```

As a CLI:

```bash
docker run --rm -v "$PWD:/work" textlens ocr /work/scan.pdf -f markdown
docker run --rm textlens doctor
```

## GPU

Requires the NVIDIA Container Toolkit.

```bash
docker build -f deploy/docker/Dockerfile.gpu -t textlens:gpu .
docker run --gpus all -p 8000:8000 -v textlens-models:/models textlens:gpu
docker run --gpus all -v textlens-models:/models textlens:gpu models install glm-ocr
```

Pick a base image tag compatible with your driver (`--build-arg BASE=…`).

## Volumes and paths

| Path | Variable | Contents |
|---|---|---|
| `/models` | `TEXTLENS_MODELS_DIR` | model weights — mount a volume to share across containers |
| `/data` | `TEXTLENS_HOME` | config, measurements, optional disk result cache |
| `/work` | — | working directory for CLI use |

Images run as a non-root user (uid 10001); the compose file uses a read-only
root filesystem with a tmpfs `/tmp`.

## Health checks

The images define `HEALTHCHECK` against `/health`. In orchestrators use
`/health` for liveness and `/ready` for readiness (it returns 503 while
warming up or when the queue is full).

## Compose

```bash
docker compose -f deploy/docker/docker-compose.yml up                    # CPU API :8000
docker compose -f deploy/docker/docker-compose.yml --profile gpu up      # GPU API :8001
docker compose -f deploy/docker/docker-compose.yml --profile served up   # vLLM + TextLens :8002
```

The `served` profile runs PaddleOCR-VL in vLLM and a TextLens API routed to
it (`TEXTLENS_PROFILE=server`, `TEXTLENS_OPENAI_BASE_URL=http://vllm:8000/v1`).

## Build arguments

| Arg | Default | Purpose |
|---|---|---|
| `EXTRAS` | `server,documents` | pip extras |
| `PRELOAD_MODELS` | `ppocrv6-small` | models baked into the image (space-separated) |
| `PYTHON_VERSION` | 3.12 | CPU image Python |
| `BASE` | see file | GPU/Jetson base image |

## Kubernetes

`deploy/kubernetes/textlens.yaml` is a reference Deployment + Service + HPA
with probes, non-root security context and Prometheus annotations. See
[Scaling](../production/scaling.md).
