"""
textlens.server
───────────────
FastAPI REST API server module for TextLens OCR.
Enables any application or developer to spin up an OCR microservice endpoint
with simple HTTP POST requests and interactive OpenAPI documentation.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field

try:
    import fastapi
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Body
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False

from textlens.sdk import TextLens
from textlens.hardware import get_hardware_info


class OCRUrlRequest(BaseModel):
    """JSON payload schema for URL / File Path OCR requests."""
    image_url: str = Field(..., description="HTTP/HTTPS URL or absolute file path to image/PDF")
    prompt: str = Field("Text Recognition:", description="Instruction prompt to guide model output")
    max_new_tokens: int = Field(512, description="Maximum token generation limit")


def create_app(engine: Optional[TextLens] = None) -> FastAPI:
    """
    Factory function to construct a FastAPI web application instance for TextLens OCR.

    Parameters
    ----------
    engine : TextLens, optional
        Initialized TextLens instance. If None, auto-initializes on startup.

    Returns
    -------
    FastAPI
        Configured FastAPI app.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi and python-multipart are required for REST server mode. "
            "Install via `pip install fastapi uvicorn python-multipart`"
        )

    app = FastAPI(
        title="TextLens OCR API",
        description=(
            "Production-ready GLM-OCR REST endpoint server. "
            "Supports image uploads, URL document reading, table/formula extraction, and hardware introspection."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for easy cross-origin web app integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global engine reference
    _engine_holder: Dict[str, Any] = {"engine": engine}

    @app.on_event("startup")
    def startup_event():
        if _engine_holder["engine"] is None:
            print("[TextLens Server] Initializing OCR engine model...")
            _engine_holder["engine"] = TextLens()

    @app.get("/", tags=["System"])
    def root():
        """Root endpoint returning service status and documentation link."""
        return {
            "name": "TextLens OCR REST Service",
            "version": "0.1.0",
            "status": "online",
            "docs": "/docs",
            "device": _engine_holder["engine"].device if _engine_holder["engine"] else "unknown"
        }

    @app.get("/api/v1/health", tags=["System"])
    def health():
        """Health check endpoint."""
        eng = _engine_holder["engine"]
        return {
            "status": "healthy" if (eng and eng.is_loaded) else "initializing",
            "model_id": eng.model_id if eng else None,
            "device": eng.device if eng else None,
        }

    @app.get("/api/v1/hardware", tags=["System"])
    def hardware():
        """Get system GPU/CPU hardware capabilities and VRAM stats."""
        return get_hardware_info().to_dict()

    @app.post("/api/v1/ocr", tags=["OCR Endpoint"])
    async def process_ocr(
        file: Optional[UploadFile] = File(None, description="Image or PDF file upload"),
        image_url: Optional[str] = Form(None, description="Image URL or local path"),
        prompt: str = Form("Text Recognition:", description="Instruction prompt"),
        max_new_tokens: int = Form(512, description="Max token generation limit")
    ):
        """
        Main OCR REST Endpoint.
        
        Send an image/PDF file upload OR an image URL/file path.
        """
        eng: TextLens = _engine_holder["engine"]
        if not eng or not eng.is_loaded:
            raise HTTPException(status_code=503, detail="OCR model is not loaded yet")

        start_time = time.time()

        try:
            if file is not None:
                # Save uploaded file to temp file
                suffix = Path(file.filename or "image.png").suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = Path(tmp.name)

                try:
                    if suffix == ".pdf":
                        pages = eng.read_pdf(tmp_path, prompt=prompt)
                        elapsed = round(time.time() - start_time, 3)
                        return {
                            "status": "success",
                            "is_pdf": True,
                            "pages": pages,
                            "device_used": eng.device,
                            "execution_time_seconds": elapsed
                        }
                    else:
                        text = eng.read(tmp_path, prompt=prompt, max_new_tokens=max_new_tokens)
                        elapsed = round(time.time() - start_time, 3)
                        return {
                            "status": "success",
                            "is_pdf": False,
                            "text": text,
                            "device_used": eng.device,
                            "execution_time_seconds": elapsed
                        }
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

            elif image_url is not None:
                # Process URL or path
                if image_url.lower().endswith(".pdf"):
                    pages = eng.read_pdf(image_url, prompt=prompt)
                    elapsed = round(time.time() - start_time, 3)
                    return {
                        "status": "success",
                        "is_pdf": True,
                        "pages": pages,
                        "device_used": eng.device,
                        "execution_time_seconds": elapsed
                    }
                else:
                    text = eng.read(image_url, prompt=prompt, max_new_tokens=max_new_tokens)
                    elapsed = round(time.time() - start_time, 3)
                    return {
                        "status": "success",
                        "is_pdf": False,
                        "text": text,
                        "device_used": eng.device,
                        "execution_time_seconds": elapsed
                    }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide either a file upload ('file') or an image URL/path ('image_url')"
                )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/ocr/json-payload", tags=["OCR Endpoint"])
    def process_ocr_json(payload: OCRUrlRequest):
        """Alternative JSON payload OCR endpoint for URL requests."""
        eng: TextLens = _engine_holder["engine"]
        if not eng or not eng.is_loaded:
            raise HTTPException(status_code=503, detail="OCR model is not loaded yet")

        start_time = time.time()
        try:
            if payload.image_url.lower().endswith(".pdf"):
                pages = eng.read_pdf(payload.image_url, prompt=payload.prompt)
                elapsed = round(time.time() - start_time, 3)
                return {
                    "status": "success",
                    "is_pdf": True,
                    "pages": pages,
                    "device_used": eng.device,
                    "execution_time_seconds": elapsed
                }
            else:
                text = eng.read(payload.image_url, prompt=payload.prompt, max_new_tokens=payload.max_new_tokens)
                elapsed = round(time.time() - start_time, 3)
                return {
                    "status": "success",
                    "is_pdf": False,
                    "text": text,
                    "device_used": eng.device,
                    "execution_time_seconds": elapsed
                }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    engine: Optional[TextLens] = None
) -> None:
    """
    Launch the TextLens REST API server with a single function call.

    Parameters
    ----------
    host : str
        Host IP address (defaults to '0.0.0.0').
    port : int
        Port number (defaults to 8000).
    reload : bool
        Enable auto-reload on code change.
    engine : TextLens, optional
        Pre-loaded engine instance.

    Examples
    --------
    >>> import textlens
    >>> textlens.serve(port=8000)
    """
    if not UVICORN_AVAILABLE:
        raise ImportError(
            "uvicorn is required to run the server. Install via `pip install uvicorn`"
        )

    print("=" * 65)
    print("             STARTING TEXTLENS OCR REST SERVER              ")
    print("=" * 65)
    print(f" REST API Endpoint : http://{host}:{port}/api/v1/ocr")
    print(f" Interactive Docs  : http://{host}:{port}/docs")
    print(f" OpenAPI Spec      : http://{host}:{port}/openapi.json")
    print("=" * 65 + "\n")

    app = create_app(engine=engine)
    uvicorn.run(app, host=host, port=port, reload=reload)
