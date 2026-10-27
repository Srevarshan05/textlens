# syntax=docker/dockerfile:1.7
# TextLens — CPU image (server + CLI). Small: no PyTorch, no CUDA.
#
#   docker build -f deploy/docker/Dockerfile -t textlens .
#   docker run -p 8000:8000 -v textlens-models:/models textlens                 # REST API
#   docker run --rm -v "$PWD:/work" textlens ocr /work/scan.pdf -f markdown      # CLI
#
# Build args:
#   PRELOAD_MODELS  space-separated model ids baked into the image (default: ppocrv6-small)
#   EXTRAS          pip extras to install (default: server,documents)

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY textlens ./textlens
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /wheels

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG EXTRAS=server,documents
ARG PRELOAD_MODELS=ppocrv6-small
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TEXTLENS_HOME=/data \
    TEXTLENS_MODELS_DIR=/models \
    TEXTLENS_HOST=0.0.0.0 \
    TEXTLENS_PORT=8000 \
    TEXTLENS_LOG_FORMAT=json

RUN useradd --create-home --uid 10001 textlens \
 && mkdir -p /data /models /work && chown -R textlens:textlens /data /models /work
COPY --from=build /wheels /wheels
RUN WHEEL="$(ls /wheels/*.whl)" && pip install --no-cache-dir "${WHEEL}[${EXTRAS}]" && rm -rf /wheels

USER textlens
# Bake verified model weights into the image so containers start offline.
RUN for m in ${PRELOAD_MODELS}; do textlens models install "$m"; done
WORKDIR /work
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"TEXTLENS_PORT\",\"8000\")}/health',timeout=4)" || exit 1
ENTRYPOINT ["textlens"]
CMD ["serve"]
