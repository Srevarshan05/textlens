"""
textlens.serving.openai_api
───────────────────────────
OpenAI-compatible facade: use TextLens from any OpenAI SDK client.

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="…")
    reply = client.chat.completions.create(
        model="textlens",                       # or a specific model id / profile:NAME
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}},
            {"type": "text", "text": "markdown"},   # optional: text | markdown
        ]}],
    )
    print(reply.choices[0].message.content)

Only ``data:`` URLs are accepted unless the server allows URL inputs.  The
``model`` field selects routing: ``textlens`` / ``auto`` (router),
``profile:<name>`` or any registered model id.
"""

# FastAPI resolves route annotations at runtime: no postponed annotations here.

import base64
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from textlens.errors import InputError


def _images_and_prompt(messages: List[Dict[str, Any]]) -> Tuple[List[str], str]:
    urls: List[str] = []
    prompt = ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            prompt = content
            continue
        for part in content or []:
            if part.get("type") == "image_url":
                url = part.get("image_url")
                urls.append(url.get("url") if isinstance(url, dict) else url)
            elif part.get("type") == "text":
                prompt = part.get("text", "")
    return urls, prompt


def register_openai_routes(app: Any, svc: Any, settings: Any, resolve_bytes: Callable[[bytes], Any]) -> None:
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.get("/v1/models", tags=["openai"])
    def list_models() -> Dict[str, Any]:
        from textlens.models.specs import all_specs

        ids = ["textlens"] + [f"profile:{p}" for p in ("edge", "fast", "balanced", "accurate", "document", "server")]
        ids += [s.id for s in all_specs() if s.runnable_locally or s.status == "remote"]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "textlens"} for i in ids]}

    @app.post("/v1/chat/completions", tags=["openai"])
    async def chat_completions(request: Request) -> Any:
        import asyncio

        body = await request.json()
        urls, prompt = _images_and_prompt(body.get("messages") or [])
        if not urls:
            raise InputError("No image_url content part found in the user message.")
        name = str(body.get("model") or "textlens")
        profile: Optional[str] = None
        model: Optional[str] = None
        if name.startswith("profile:"):
            profile = name.split(":", 1)[1]
        elif name not in ("textlens", "auto"):
            model = name
        task = "markdown" if "markdown" in prompt.lower() else "text"
        sources = []
        for url in urls:
            if url.startswith("data:"):
                try:
                    payload = base64.b64decode(url.split(",", 1)[1], validate=False)
                except Exception as exc:
                    raise InputError("Malformed data URL.") from exc
                sources.append(resolve_bytes(payload))
            elif settings.allow_urls:
                from textlens.inputs.source import default_url_guard, open_source

                sources.append(await asyncio.to_thread(open_source, url, allow_urls=True, url_guard=default_url_guard,
                                                      max_bytes=settings.max_upload_mb * 1024 * 1024))
            else:
                raise InputError("Remote image URLs are disabled; send a base64 data: URL.")
        engine = svc.engine(profile, model)

        def work(job: Any) -> List[Any]:
            return [engine(src, task=task, max_pages=settings.max_pages) for src in sources]

        job = svc.jobs.submit(work, kind="openai", meta={"images": len(sources)})
        fut = svc.jobs.future(job.id)
        if fut is not None:
            await asyncio.wrap_future(fut)
        job = svc.jobs.get(job.id)
        if job.status != "succeeded":
            return JSONResponse({"error": {"message": (job.error or {}).get("message", "failed"), "type": "server_error"}}, status_code=500)
        results = job.result
        content = "\n\n".join(r.to_markdown() if task == "markdown" else r.text for r in results)
        confidence = [r.confidence for r in results if r.confidence is not None]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ",".join(sorted({m for r in results for m in r.provenance.models})) or name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "textlens": {"confidence": min(confidence) if confidence else None, "pages": sum(r.page_count for r in results)},
        }
