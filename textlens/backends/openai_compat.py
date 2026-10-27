"""
textlens.backends.openai_compat
───────────────────────────────
Remote OCR through any OpenAI-compatible chat-completions endpoint —
vLLM, SGLang, LMDeploy, TGI, Ollama, or a hosted gateway.

This is TextLens' **server tier**: heavyweight document VLMs (PaddleOCR-VL,
DeepSeek-OCR, GLM-OCR, …) run in an inference server with continuous
batching, while TextLens keeps routing, selective OCR, validation and the
unified result schema.  Pages are sent concurrently so the server can batch
them.

Configuration (argument > environment)
    endpoint    ``TEXTLENS_OPENAI_BASE_URL`` or ``OPENAI_BASE_URL``
                (e.g. ``http://localhost:8000/v1``)
    api_key     ``TEXTLENS_OPENAI_API_KEY`` or ``OPENAI_API_KEY``
    served_model  name the server knows the model by (defaults to the
                spec's Hugging Face repo id, which is what vLLM uses)

Privacy: remote backends are never chosen automatically unless an endpoint
is configured — documents leave the machine only when you opt in.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from textlens.backends.base import OCRBackend, PageOCR, RecognizeOptions
from textlens.errors import BackendUnavailableError, InferenceError

logger = logging.getLogger("textlens.backends.openai")

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def configured_endpoint(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get("TEXTLENS_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


class OpenAICompatibleBackend(OCRBackend):
    """Send page images to an OpenAI-compatible ``/chat/completions`` API.

    Options
    -------
    endpoint, api_key, served_model : str
    concurrency : int      parallel page requests (default 8)
    timeout : float        seconds per request (default 180)
    max_retries : int      retries on 429/5xx with jittered backoff (default 3)
    max_side : int         downscale images so the longest side ≤ this (default 1800)
    image_format : str     "png" (lossless, default) or "jpeg"
    logprobs : bool        request token logprobs for a confidence proxy (default True)
    """

    thread_safe = True
    backend_name = "openai-compatible"

    def _load(self) -> None:
        endpoint = configured_endpoint(self.options.get("endpoint"))
        if not endpoint:
            raise BackendUnavailableError(
                f"No inference endpoint configured for {self.spec.id}.",
                hint="Pass endpoint='http://host:8000/v1' or set TEXTLENS_OPENAI_BASE_URL (e.g. a vLLM server).",
            )
        self._endpoint = endpoint.rstrip("/")
        self._api_key = self.options.get("api_key") or os.environ.get("TEXTLENS_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._model = self.options.get("served_model") or self.spec.hf_repo_id or self.spec.id
        self.device = f"remote:{self._endpoint}"

    # ── HTTP ─────────────────────────────────────────────────────────────
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._endpoint}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        retries = int(self.options.get("max_retries", 3))
        timeout = float(self.options.get("timeout", 180.0))
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured endpoint
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last = InferenceError(f"{self._endpoint} returned HTTP {exc.code}: {detail}")
                if exc.code == 400 and payload.get("logprobs"):
                    # Some servers reject logprobs; retry once without them.
                    payload = dict(payload)
                    payload.pop("logprobs", None)
                    body = json.dumps(payload).encode("utf-8")
                    continue
                if exc.code not in _RETRY_STATUS:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = InferenceError(f"Cannot reach {self._endpoint}: {exc}")
            if attempt < retries:
                time.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise last or InferenceError("request failed")

    def _encode(self, img: Any) -> str:
        from textlens.inputs.images import limit_size

        img, _ = limit_size(img, int(self.options.get("max_side", 1800)))
        fmt = str(self.options.get("image_format", "png")).lower()
        buf = io.BytesIO()
        if fmt in ("jpg", "jpeg"):
            img.save(buf, format="JPEG", quality=92)
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            mime = "image/png"
        return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    def _prompt(self, options: RecognizeOptions) -> str:
        if options.prompt:
            return options.prompt
        prompts = self.spec.prompts or {}
        return prompts.get(options.task) or prompts.get("markdown") or prompts.get("text") or (
            "Convert this document page to Markdown. Preserve reading order, headings, lists and tables "
            "(as Markdown tables) and write formulas in LaTeX. Output only the content."
        )

    def _one(self, img: Any, options: RecognizeOptions) -> PageOCR:
        t0 = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._encode(img)}},
                        {"type": "text", "text": self._prompt(options)},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": int(options.max_new_tokens or self.options.get("max_tokens", 4096)),
        }
        if self.options.get("logprobs", True):
            payload["logprobs"] = True
        data = self._post(payload)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"Unexpected response from {self._endpoint}: {str(data)[:300]}") from exc
        confidence = None
        tokens = ((choice.get("logprobs") or {}).get("content")) or []
        if tokens:
            probs = [math.exp(t["logprob"]) for t in tokens if isinstance(t, dict) and t.get("logprob") is not None]
            if probs:
                confidence = round(sum(probs) / len(probs), 4)
        warnings = []
        if choice.get("finish_reason") == "length":
            warnings.append("output truncated at max_tokens")
        return PageOCR(
            width=img.size[0],
            height=img.size[1],
            markdown=text.strip(),
            confidence=confidence,
            timings_ms={"request": round((time.perf_counter() - t0) * 1000, 1)},
            warnings=warnings,
            raw={"usage": data.get("usage")},
        )

    def _recognize(self, images: List[Any], options: RecognizeOptions) -> List[PageOCR]:
        if len(images) == 1:
            return [self._one(images[0], options)]
        workers = max(1, min(int(self.options.get("concurrency", 8)), len(images)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="textlens-remote") as pool:
            return list(pool.map(lambda im: self._one(im, options), images))
