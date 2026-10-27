"""
textlens.serving.legacy
───────────────────────
TextLens 0.x REST routes, rebuilt on the 2.0 engine.

``/api/v1/health``, ``/api/v1/hardware``, ``/api/v1/ocr`` and
``/api/v1/ocr/json-payload`` keep their request and response shapes.

Security fix vs 0.x: ``image_url`` used to accept *any* local file path,
letting a client read arbitrary files from the server.  Now only
``http(s)`` URLs are accepted (when the server allows URLs, with an SSRF
guard), and local paths only inside ``TEXTLENS_ALLOWED_ROOT`` when set.
"""

# FastAPI resolves route annotations at runtime: no postponed annotations here.

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional

from textlens.errors import InputError


def register_legacy_routes(app: Any, svc: Any, settings: Any, read_upload: Any) -> None:
    from fastapi import File, Form, UploadFile
    from pydantic import BaseModel, Field

    class OCRUrlRequest(BaseModel):
        image_url: str = Field(..., description="HTTP/HTTPS URL (or a path under TEXTLENS_ALLOWED_ROOT)")
        prompt: str = Field("Text Recognition:")
        max_new_tokens: int = Field(512)

    def open_ref(ref: str) -> Any:
        from textlens.inputs.source import default_url_guard, open_source

        max_bytes = settings.max_upload_mb * 1024 * 1024
        if ref.lower().startswith(("http://", "https://")):
            if not settings.allow_urls:
                raise InputError("URL inputs are disabled on this server.", hint="Start the server with --allow-urls.")
            return open_source(ref, allow_urls=True, max_bytes=max_bytes, url_guard=default_url_guard)
        if not settings.allowed_root:
            raise InputError(
                "Local file paths are not accepted.",
                hint="Upload the file instead, or set TEXTLENS_ALLOWED_ROOT to a directory the API may read.",
            )
        return open_source(ref, allowed_root=Path(settings.allowed_root), max_bytes=max_bytes)

    def legacy_response(result: Any, t0: float, device: Optional[str]) -> Dict[str, Any]:
        elapsed = round(time.time() - t0, 3)
        if result.metadata.get("kind") == "pdf":
            total = result.metadata.get("document_page_count", result.page_count)
            pages = [{"page": p.number, "total_pages": total, "text": p.text} for p in result.pages]
            return {"status": "success", "is_pdf": True, "pages": pages, "device_used": device, "execution_time_seconds": elapsed}
        return {"status": "success", "is_pdf": False, "text": result.text, "device_used": device, "execution_time_seconds": elapsed}

    def run(src: Any, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        t0 = time.time()
        kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if prompt and prompt != "Text Recognition:":
            kwargs["prompt"] = prompt
        result = svc.engine()(src, **kwargs)
        device = next((p.provenance.device for p in result.pages if p.provenance.device), None)
        return legacy_response(result, t0, device)

    @app.get("/api/v1/health", tags=["legacy"])
    def legacy_health() -> Dict[str, Any]:
        leg = svc.legacy_engine
        if leg is not None:
            return {
                "status": "healthy" if getattr(leg, "is_loaded", False) else "initializing",
                "model_id": getattr(leg, "model_id", None),
                "device": getattr(leg, "device", None),
                "is_cuda": leg.is_cuda() if hasattr(leg, "is_cuda") else False,
            }
        return {"status": "healthy", "model_id": None, "device": None, "is_cuda": False}

    @app.get("/api/v1/hardware", tags=["legacy"])
    def legacy_hardware() -> Dict[str, Any]:
        from textlens.hardware import get_hardware_info

        return get_hardware_info().to_dict()

    @app.post("/api/v1/ocr", tags=["legacy"])
    async def legacy_ocr(
        file: Optional[UploadFile] = File(None),
        image_url: Optional[str] = Form(None),
        prompt: str = Form("Text Recognition:"),
        max_new_tokens: int = Form(512),
    ) -> Any:
        if file is not None:
            src = await read_upload(file)
        elif image_url:
            src = await asyncio.to_thread(open_ref, image_url)
        else:
            raise InputError("Must provide either a file upload ('file') or an image URL ('image_url').")
        return await asyncio.to_thread(run, src, prompt, max_new_tokens)

    @app.post("/api/v1/ocr/json-payload", tags=["legacy"])
    async def legacy_ocr_json(payload: OCRUrlRequest) -> Any:
        src = await asyncio.to_thread(open_ref, payload.image_url)
        return await asyncio.to_thread(run, src, payload.prompt, payload.max_new_tokens)
