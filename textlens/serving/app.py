"""
textlens.serving.app
────────────────────
Production REST API for TextLens (``pip install "textlens-ocr[server]"``).

Endpoints
    POST /ocr            image/PDF → Result (sync, ``?async=true`` → job, ``?stream=true`` → NDJSON pages)
    POST /document       structure-first processing (Markdown task)
    POST /extract        schema → JSON fields
    POST /inspect        PDF classification / image signals (no OCR, cheap)
    POST /batch          many files → async job
    GET  /jobs/{id}      job status · GET /jobs/{id}/result · DELETE /jobs/{id} (cancel)
    GET  /models         catalog with install/runtime status
    GET  /health         liveness          GET /ready   readiness (model routable, queue not full)
    GET  /metrics        Prometheus text   GET /system  hardware + runtimes
    POST /v1/chat/completions, GET /v1/models   OpenAI-compatible OCR facade
    /api/v1/*            0.x endpoints (compatibility)

Design
    * Uploads stay in memory (bounded by ``max_upload_mb``); nothing is
      written to disk, so ``no_persist`` holds by construction.
    * Inference runs in a bounded :class:`~textlens.runtime.jobs.JobManager`;
      when the queue is full requests get ``503`` + ``Retry-After`` instead
      of exhausting GPU memory (backpressure).
    * The same app scales out: put it behind a gateway, set ``workers`` per
      GPU, and swap the job store for a shared one to split API and workers.
"""

# FastAPI resolves route annotations at runtime: no postponed annotations here.

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from textlens.errors import CapacityError, InputError, InputTooLargeError, TextLensError
from textlens.observability import metrics
from textlens.serving.security import SECURITY_HEADERS, AuthHook, RateLimiter, api_key_principal
from textlens.serving.settings import ServerSettings

logger = logging.getLogger("textlens.server")

_http_requests = metrics.counter("textlens_http_requests_total", "HTTP requests")
_http_seconds = metrics.histogram("textlens_http_request_seconds", "HTTP request latency")
_pages_total = metrics.counter("textlens_pages_total", "Pages processed by source")

_PUBLIC_PATHS = {"/health", "/ready", "/docs", "/redoc", "/openapi.json", "/", "/docs/oauth2-redirect"}


def _require_fastapi() -> Any:
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:
        from textlens.errors import BackendUnavailableError

        raise BackendUnavailableError("The server needs FastAPI.", hint='pip install "textlens-ocr[server]"') from exc
    return fastapi


class _Service:
    """Engine cache + job manager shared by all routes."""

    def __init__(self, settings: ServerSettings, ocr: Any = None, legacy_engine: Any = None) -> None:
        self.settings = settings
        self._engines: Dict[tuple, Any] = {}
        self._lock = threading.Lock()
        if ocr is not None:
            self._engines[(None, None)] = ocr
        self.legacy_engine = legacy_engine
        self.jobs = _make_jobs(settings)
        self.started = time.time()
        self.warm = not settings.warmup

    def engine(self, profile: Optional[str] = None, model: Optional[str] = None) -> Any:
        from textlens.core.engine import OCR

        key = (profile, model)
        with self._lock:
            if key not in self._engines:
                if len(self._engines) > 16:  # bound per-request override combos
                    self._engines.pop(next(iter(self._engines)))
                self._engines[key] = OCR(
                    model=model or self.settings.model,
                    profile=profile or self.settings.profile,
                    device=self.settings.device,
                    endpoint=self.settings.endpoint,
                    cache="off" if self.settings.no_persist else "memory",
                )
            return self._engines[key]


def _make_jobs(settings: ServerSettings) -> Any:
    from textlens.runtime.jobs import JobManager

    return JobManager(workers=settings.workers, max_queue=settings.max_queue, ttl_seconds=settings.job_ttl)


def create_app(
    settings: Optional[ServerSettings] = None,
    ocr: Any = None,
    auth: Optional[AuthHook] = None,
    engine: Any = None,
) -> Any:
    """Build the FastAPI application.

    Parameters
    ----------
    settings:
        :class:`ServerSettings` (defaults from environment variables).
    ocr:
        A pre-configured :class:`~textlens.OCR` to serve (optional).
    auth:
        ``callable(request) -> principal | None`` for custom authentication
        (JWT, OIDC, mTLS headers from a gateway…).  Overrides API keys.
    engine:
        0.x ``TextLens`` instance for the legacy ``/api/v1`` routes.
    """
    _require_fastapi()
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse

    from textlens import __version__

    settings = settings or ServerSettings.from_env()
    svc = _Service(settings, ocr=ocr, legacy_engine=engine)
    limiter = RateLimiter(settings.rate_limit, settings.rate_burst)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    async def lifespan(app: Any):  # type: ignore[no-untyped-def]
        if settings.warmup:
            await asyncio.to_thread(lambda: svc.engine().warmup())
            svc.warm = True
        yield
        svc.jobs.shutdown(wait=False)

    from contextlib import asynccontextmanager

    app = FastAPI(
        title="TextLens OCR API",
        version=__version__,
        description="Model-agnostic OCR and document intelligence. Selective OCR, routed models, unified results.",
        lifespan=asynccontextmanager(lifespan),
    )
    app.state.textlens = svc
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

    # ── middleware: request id, auth, rate limit, headers, metrics ──────
    @app.middleware("http")
    async def gate(request: Request, call_next: Callable) -> Any:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        path = request.url.path
        public = path in _PUBLIC_PATHS or (path == "/metrics" and settings.metrics_public)
        principal = request.client.host if request.client else "anonymous"
        if not public and (auth is not None or settings.api_keys):
            principal = auth(request) if auth is not None else api_key_principal(request, settings.api_keys)
            if principal is None:
                return JSONResponse({"error": "unauthorized", "message": "Missing or invalid API key."}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer", "X-Request-ID": rid})
        if not public:
            allowed, retry = limiter.allow(str(principal))
            if not allowed:
                return JSONResponse({"error": "rate_limited", "message": "Too many requests."}, status_code=429,
                                    headers={"Retry-After": f"{max(1, int(retry + 0.999))}", "X-Request-ID": rid})
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes * max(1, settings.batch_max_files if path == "/batch" else 1) + 1_000_000:
            return JSONResponse({"error": "input_too_large", "message": f"Request exceeds {settings.max_upload_mb} MB."}, status_code=413)
        request.state.principal = principal
        request.state.request_id = rid
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        response.headers["X-Request-ID"] = rid
        route = request.scope.get("route")
        label = getattr(route, "path", path)
        _http_requests.inc(route=label, method=request.method, status=str(response.status_code))
        _http_seconds.observe(time.perf_counter() - t0, route=label)
        logger.info("request", extra={"request_id": rid, "path": path, "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return response

    @app.exception_handler(TextLensError)
    async def textlens_error(request: Request, exc: TextLensError) -> Any:
        headers = {}
        if isinstance(exc, CapacityError):
            headers["Retry-After"] = str(exc.details.get("retry_after", 5))
        return JSONResponse(exc.to_dict(), status_code=exc.http_status, headers=headers)

    # ── helpers ──────────────────────────────────────────────────────────
    async def read_upload(upload: Any) -> Any:
        from textlens.inputs.source import open_source

        data = await upload.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise InputTooLargeError(f"{upload.filename or 'upload'} exceeds {settings.max_upload_mb} MB.")
        return open_source(data, name=upload.filename or "upload", max_bytes=max_bytes)

    async def resolve(file: Any, url: Optional[str]) -> Any:
        from textlens.inputs.source import default_url_guard, open_source

        if file is not None:
            return await read_upload(file)
        if url:
            if not settings.allow_urls:
                raise InputError("URL inputs are disabled on this server.", hint="Upload the file, or start the server with --allow-urls.")
            if not url.lower().startswith(("http://", "https://")):
                raise InputError("Only http(s) URLs are accepted.")
            return await asyncio.to_thread(open_source, url, allow_urls=True, max_bytes=max_bytes, url_guard=default_url_guard)
        raise InputError("Provide a file upload ('file') or a 'url'.")

    async def run_job(fn: Callable[[Any], Any], kind: str, meta: Dict[str, Any]) -> Any:
        job = svc.jobs.submit(fn, kind=kind, meta=meta)
        fut = svc.jobs.future(job.id)
        if fut is not None:
            try:
                await asyncio.wait_for(asyncio.wrap_future(fut), timeout=settings.sync_timeout)
            except asyncio.TimeoutError:
                return JSONResponse({"error": "timeout", "message": "Still processing; poll the job.", "job_id": job.id}, status_code=202)
        job = svc.jobs.get(job.id)
        if job.status == "failed":
            status = 422 if (job.error or {}).get("error", "").startswith(("input", "unsupported", "document", "encrypted")) else 500
            return JSONResponse(dict(job.error or {}, job_id=job.id), status_code=status)
        if job.status == "cancelled":
            return JSONResponse({"error": "cancelled", "job_id": job.id}, status_code=409)
        return job.result

    def respond(result: Any, fmt: str) -> Any:
        fmt = (fmt or "json").lower()
        if fmt == "text":
            return PlainTextResponse(result.text)
        if fmt in ("markdown", "md"):
            return PlainTextResponse(result.to_markdown(page_markers=result.page_count > 1), media_type="text/markdown; charset=utf-8")
        if fmt == "html":
            return HTMLResponse(result.to_html())
        body = result.to_dict()
        body["text"] = result.text
        body["confidence"] = result.confidence
        return JSONResponse(body)

    def parse_pages(pages: Optional[str]) -> Optional[List[int]]:
        from textlens.cli.output import parse_pages as pp

        try:
            return pp(pages)
        except ValueError as exc:
            raise InputError(f"Invalid pages selection {pages!r}.") from exc

    def count_pages(result: Any) -> None:
        for p in result.pages:
            _pages_total.inc(source=p.provenance.source)

    # ── system ───────────────────────────────────────────────────────────
    @app.get("/", tags=["system"])
    def root() -> Dict[str, Any]:
        info: Dict[str, Any] = {"name": "TextLens OCR REST Service", "version": __version__, "status": "online", "docs": "/docs"}
        leg = svc.legacy_engine
        if leg is not None:  # 0.x fields
            info["device"] = getattr(leg, "device", None)
            info["cuda_accelerated"] = leg.is_cuda() if hasattr(leg, "is_cuda") else False
        return info

    @app.get("/health", tags=["system"])
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - svc.started, 1)}

    @app.get("/ready", tags=["system"])
    def ready() -> Any:
        problems = []
        if not svc.warm:
            problems.append("warming up")
        if svc.jobs.saturated:
            problems.append("queue full")
        try:
            decision = svc.engine().router.route()
        except TextLensError as exc:
            problems.append(exc.message)
            decision = None
        body = {"ready": not problems, "problems": problems, "default_model": decision.model if decision else None, "jobs": svc.jobs.stats()}
        return JSONResponse(body, status_code=200 if not problems else 503)

    @app.get("/metrics", tags=["system"])
    def metrics_endpoint() -> Any:
        from textlens.backends.loader import get_pool

        g = metrics.gauge("textlens_models_loaded", "Loaded model backends")
        g.set(len(get_pool().loaded()))
        metrics.gauge("textlens_queue_pending", "Jobs queued or running").set(svc.jobs.pending)
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/system", tags=["system"])
    def system() -> Dict[str, Any]:
        from textlens.backends.loader import get_pool
        from textlens.runtime.system import inspect_system

        return {
            "system": inspect_system().to_dict(),
            "loaded_models": [b.describe() for b in get_pool().loaded()],
            "settings": settings.public_dict(),
            "jobs": svc.jobs.stats(),
        }

    @app.get("/models", tags=["models"])
    def models() -> List[Dict[str, Any]]:
        from textlens.backends.loader import missing_requirements
        from textlens.models import artifacts
        from textlens.models.specs import all_specs

        out = []
        for s in all_specs():
            out.append({
                "id": s.id, "name": s.display_name, "backend": s.backend, "status": s.status, "tasks": sorted(s.tasks),
                "license": s.license, "commercial_use": s.commercial_use, "min_vram_gb": s.min_vram_gb, "edge": s.edge,
                "installed": artifacts.is_installed(s) if s.runnable_locally else None,
                "runtime_ready": not missing_requirements(s) if s.runnable_locally else None,
            })
        return out

    # ── OCR ──────────────────────────────────────────────────────────────
    async def _ocr_route(file: Any, url: Optional[str], pages: Optional[str], ocr_mode: str, profile: Optional[str], model: Optional[str],
                         fmt: str, run_async: bool, stream: bool, task: str) -> Any:
        src = await resolve(file, url)
        engine_ = svc.engine(profile, model)
        opts: Dict[str, Any] = {"pages": parse_pages(pages), "ocr": ocr_mode, "task": task, "max_pages": settings.max_pages}
        meta = {"source": src.name, "bytes": src.size, "mime": src.mime}

        if stream:
            q: "queue.Queue[Any]" = queue.Queue(maxsize=16)
            done = object()

            def produce(job: Any) -> Dict[str, Any]:
                n = 0
                try:
                    for page in engine_.stream(src, **opts):
                        job.check_cancelled()
                        n += 1
                        job.progress = {"pages_done": n}
                        _pages_total.inc(source=page.provenance.source)
                        q.put(json.dumps({"type": "page", "page": _page_dict(page)}, ensure_ascii=False))
                    q.put(json.dumps({"type": "done", "pages": n}))
                except Exception as exc:  # delivered in-band
                    q.put(json.dumps({"type": "error", "error": getattr(exc, "code", "error"), "message": getattr(exc, "message", str(exc))}))
                finally:
                    q.put(done)
                return {"pages": n}

            svc.jobs.submit(produce, kind="stream", meta=meta)

            async def gen() -> Any:
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item is done:
                        break
                    yield item + "\n"

            return StreamingResponse(gen(), media_type="application/x-ndjson")

        def work(job: Any) -> Any:
            def events(event: str, payload: Dict[str, Any]) -> None:
                job.check_cancelled()
                if event == "document_start":
                    job.progress = {"pages_total": payload.get("pages"), "pages_done": 0}
                elif event == "page_done":
                    job.progress["pages_done"] = job.progress.get("pages_done", 0) + 1

            result = engine_(src, events=events, **opts)
            count_pages(result)
            return result

        if run_async:
            job = svc.jobs.submit(lambda j: _serialise(work(j)), kind="ocr", meta=dict(meta, format="json"))
            return JSONResponse({"job_id": job.id, "status": job.status, "status_url": f"/jobs/{job.id}"}, status_code=202)
        result = await run_job(work, "ocr", meta)
        if isinstance(result, Response):
            return result
        return respond(result, fmt)

    @app.post("/ocr", tags=["ocr"])
    async def ocr_endpoint(
        file: Optional[UploadFile] = File(None, description="Image, PDF, DOCX or PPTX"),
        url: Optional[str] = Form(None, description="http(s) URL (if the server allows URLs)"),
        pages: Optional[str] = Form(None, description="Page selection, e.g. 1-3,7"),
        ocr: str = Form("auto", description="auto (selective) | force | off"),
        profile: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        format: str = Form("json", description="json | text | markdown | html"),
        run_async: bool = Form(False, alias="async"),
        stream: bool = Form(False, description="NDJSON page stream"),
    ) -> Any:
        return await _ocr_route(file, url, pages, ocr, profile, model, format, run_async, stream, "text")

    @app.post("/document", tags=["ocr"])
    async def document_endpoint(
        file: Optional[UploadFile] = File(None),
        url: Optional[str] = Form(None),
        pages: Optional[str] = Form(None),
        ocr: str = Form("auto"),
        profile: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        format: str = Form("markdown"),
        run_async: bool = Form(False, alias="async"),
        stream: bool = Form(False),
    ) -> Any:
        return await _ocr_route(file, url, pages, ocr, profile, model, format, run_async, stream, "markdown")

    @app.post("/inspect", tags=["ocr"])
    async def inspect_endpoint(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None), sample: Optional[int] = Form(None)) -> Any:
        src = await resolve(file, url)
        report = await asyncio.to_thread(lambda: svc.engine().inspect(src, sample=sample))
        return report.to_dict() if hasattr(report, "to_dict") else report

    @app.post("/extract", tags=["ocr"])
    async def extract_endpoint(
        file: Optional[UploadFile] = File(None),
        url: Optional[str] = Form(None),
        schema_: str = Form(..., alias="schema", description='JSON mapping, e.g. {"total": "float"}, JSON Schema, or comma-separated names'),
        strategy: str = Form("auto"),
        profile: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
    ) -> Any:
        src = await resolve(file, url)
        try:
            parsed: Any = json.loads(schema_) if schema_.strip().startswith(("{", "[")) else [s.strip() for s in schema_.split(",") if s.strip()]
        except ValueError as exc:
            raise InputError("schema is not valid JSON.") from exc
        engine_ = svc.engine(profile, model)
        out = await run_job(lambda job: engine_.extract(src, schema=parsed, strategy=strategy), "extract", {"source": src.name})
        if isinstance(out, Response):
            return out
        return out.to_dict()

    @app.post("/batch", tags=["jobs"], status_code=202)
    async def batch_endpoint(
        files: List[UploadFile] = File(...),
        ocr: str = Form("auto"),
        profile: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
    ) -> Any:
        if len(files) > settings.batch_max_files:
            raise InputError(f"At most {settings.batch_max_files} files per batch.")
        sources = [await read_upload(f) for f in files]
        engine_ = svc.engine(profile, model)

        def work(job: Any) -> Dict[str, Any]:
            items = []
            job.progress = {"files_total": len(sources), "files_done": 0}
            for src in sources:
                job.check_cancelled()
                try:
                    res = engine_(src, ocr=ocr, max_pages=settings.max_pages)
                    count_pages(res)
                    items.append({"source": src.name, "status": "succeeded", "result": _serialise(res)})
                except TextLensError as exc:
                    items.append({"source": src.name, "status": "failed", "error": exc.to_dict()})
                job.progress["files_done"] += 1
            return {"items": items}

        job = svc.jobs.submit(work, kind="batch", meta={"files": [s.name for s in sources]})
        return {"job_id": job.id, "status": job.status, "status_url": f"/jobs/{job.id}", "files": len(sources)}

    @app.get("/jobs/{job_id}", tags=["jobs"])
    def job_status(job_id: str) -> Dict[str, Any]:
        return svc.jobs.get(job_id).to_dict()

    @app.get("/jobs/{job_id}/result", tags=["jobs"])
    def job_result(job_id: str) -> Any:
        job = svc.jobs.get(job_id)
        if job.status in ("queued", "running"):
            return JSONResponse({"status": job.status, "progress": job.progress}, status_code=202)
        if job.status != "succeeded":
            return JSONResponse(job.to_dict(), status_code=409)
        return JSONResponse(job.result if isinstance(job.result, dict) else _serialise(job.result))

    @app.delete("/jobs/{job_id}", tags=["jobs"])
    def job_cancel(job_id: str) -> Dict[str, Any]:
        return svc.jobs.cancel(job_id).to_dict()

    # ── OpenAI-compatible facade ─────────────────────────────────────────
    if settings.openai_api:
        from textlens.serving.openai_api import register_openai_routes

        register_openai_routes(app, svc, settings, resolve_bytes=lambda b: _source_from_bytes(b, max_bytes))

    # ── 0.x compatibility routes ─────────────────────────────────────────
    from textlens.serving.legacy import register_legacy_routes

    register_legacy_routes(app, svc, settings, read_upload)
    return app


def _source_from_bytes(data: bytes, max_bytes: int) -> Any:
    from textlens.inputs.source import open_source

    return open_source(data, name="image", max_bytes=max_bytes)


def _page_dict(page: Any) -> Dict[str, Any]:
    from textlens.core.result import _to_jsonable

    d = _to_jsonable(page)
    d["text"] = page.text
    return d


def _serialise(result: Any) -> Dict[str, Any]:
    body = result.to_dict()
    body["text"] = result.text
    body["confidence"] = result.confidence
    return body


def serve(settings: Optional[ServerSettings] = None, host: Optional[str] = None, port: Optional[int] = None, reload: bool = False, engine: Any = None, **overrides: Any) -> None:
    """Run the server with uvicorn (``textlens serve``)."""
    _require_fastapi()
    try:
        import uvicorn
    except ImportError as exc:
        from textlens.errors import BackendUnavailableError

        raise BackendUnavailableError("The server needs uvicorn.", hint='pip install "textlens-ocr[server]"') from exc
    from textlens import __version__
    from textlens.observability import configure_logging

    settings = settings or ServerSettings.from_env()
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if overrides:
        settings = settings.replace(**overrides)
    configure_logging(level=logging.INFO)
    app = create_app(settings, engine=engine)
    shown = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    print(f"TextLens {__version__} API  →  http://{shown}:{settings.port}   (docs: /docs, health: /health, metrics: /metrics)")
    if settings.host in ("0.0.0.0", "::") and not settings.api_keys:
        print("WARNING: listening on all interfaces without API keys. Set TEXTLENS_API_KEYS or --api-key.")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
