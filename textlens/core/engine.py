"""
textlens.core.engine
────────────────────
``OCR`` — the one TextLens API, from a Raspberry Pi to a GPU cluster.

    >>> from textlens import OCR
    >>> print(OCR()("invoice.pdf").text)

Beginners pass a file; TextLens inspects it, extracts native text where it
can, OCRs only what needs it, picks a model for the hardware and returns a
:class:`~textlens.core.result.Result`.  Power users pin models, backends,
devices, budgets and fallback chains — the call site does not change.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from textlens.core.pipeline import Pipeline, PipelineOptions
from textlens.core.result import Page, Result, RunProvenance
from textlens.core.router import PageSignals, Router, RoutingDecision

logger = logging.getLogger("textlens.ocr")

PageSpec = Union[int, Sequence[int], None]


def _pages_arg(pages: PageSpec) -> Optional[List[int]]:
    if pages is None:
        return None
    if isinstance(pages, int):
        return [pages]
    return [int(p) for p in pages]


class OCR:
    """Unified OCR and document-intelligence engine.

    Parameters
    ----------
    model : str, optional
        Pin a model (``"ppocrv6-small"``, ``"glm-ocr"``, a served model id…).
        Omit to let the router choose per page.
    device : str, optional
        ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"tensorrt"``, ``"coreml"`` …
    auto_download : bool
        Download missing models on first use (the small default model is
        always allowed; large models only when explicitly requested).
    profile : str
        ``auto`` (default), ``edge``, ``fast``, ``balanced``, ``accurate``,
        ``document`` or ``server``.
    backend : str, optional
        ``"openai"`` to run ``model`` on an OpenAI-compatible server.
    endpoint, api_key : str, optional
        Remote inference server (vLLM, SGLang, …) for ``backend="openai"``.
    fallback : bool, optional
        Retry low-confidence pages with the next model (profile default).
    fallback_models : list[str], optional
        Explicit fallback chain.
    min_confidence : float, optional
        Accept threshold for a page before falling back.
    latency_budget_ms, memory_budget_gb : float, optional
        Budgets the router must respect when choosing models.
    quality : "high" | "low", optional
        Shorthand that nudges the ``auto`` profile.
    ocr : "auto" | "force" | "off"
        Selective OCR (default), OCR every page, or native text only.
    dpi : int, optional
        Rasterisation resolution for PDF pages that need OCR.
    task : str
        ``"text"`` (default), ``"markdown"``, ``"table"``, ``"formula"``.
    cache : "off" | "memory" | "disk", optional
        Result cache mode (default from ``TEXTLENS_RESULT_CACHE``).
    **backend_options
        Forwarded to the backend adapter (e.g. ``threads=4``, ``low_memory=True``).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        auto_download: bool = True,
        *,
        profile: Optional[str] = None,
        backend: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback: Optional[bool] = None,
        fallback_models: Optional[Sequence[str]] = None,
        min_confidence: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        memory_budget_gb: Optional[float] = None,
        quality: Optional[str] = None,
        ocr: str = "auto",
        dpi: Optional[int] = None,
        task: str = "text",
        cache: Optional[str] = None,
        allowed_licenses: Optional[Sequence[str]] = None,
        commercial_only: bool = False,
        detect_tables: bool = True,
        ocr_images: bool = False,
        reuse_ocr_layer: Optional[bool] = None,
        password: Optional[str] = None,
        **backend_options: Any,
    ) -> None:
        if api_key:
            backend_options["api_key"] = api_key
        if not auto_download:
            backend_options["auto_download"] = False
        self._router = Router(
            profile,
            model=model,
            backend=backend,
            device=device,
            endpoint=endpoint,
            fallback=fallback,
            fallback_models=fallback_models,
            min_confidence=min_confidence,
            latency_budget_ms=latency_budget_ms,
            memory_budget_gb=memory_budget_gb,
            quality=quality,
            allow_download=auto_download,
            allowed_licenses=allowed_licenses,
            commercial_only=commercial_only,
            dpi=dpi,
            backend_options=backend_options,
        )
        self._defaults = PipelineOptions(
            ocr=ocr,
            task=task,
            detect_tables=detect_tables,
            ocr_images=ocr_images,
            reuse_ocr_layer=reuse_ocr_layer,
            password=password,
        )
        self._model = model
        self._device = device
        self._auto_download = auto_download
        self._cache_mode = cache
        # 0.x behaviour: an explicitly requested local model that is not
        # installed fails fast when downloads are disabled.
        spec = self._router._explicit_spec
        if spec is not None and not auto_download and spec.backend != "openai":
            from textlens.errors import ModelNotInstalledError
            from textlens.models import artifacts

            if not artifacts.is_installed(spec):
                raise ModelNotInstalledError(spec.id)
        logger.info("OCR ready (profile=%s, model=%s, device=%s)", self._router.profile.name, model or "auto", device or "auto")

    # ── configuration views ──────────────────────────────────────────────
    @property
    def profile(self) -> str:
        return self._router.profile.name

    @property
    def router(self) -> Router:
        return self._router

    @property
    def model_id(self) -> str:
        """Pinned model, or the model the router would choose for a plain image."""
        if self._router._explicit_spec is not None:
            return self._router._explicit_spec.id
        try:
            return self._router.route(PageSignals()).model
        except Exception:
            from textlens.models.specs import default_spec

            return default_spec().id

    @property
    def model_name(self) -> str:
        return self._router.spec_for(self.model_id).display_name

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def is_loaded(self) -> bool:
        from textlens.backends.loader import get_pool

        return any(b.spec.id == self.model_id for b in get_pool().loaded())

    def _options(self, overrides: Dict[str, Any]) -> PipelineOptions:
        opts = PipelineOptions(**{**self._defaults.__dict__})
        for key, value in overrides.items():
            if key == "page":  # 0.x keyword
                key, value = "pages", value
            if key == "pages":
                value = _pages_arg(value)
            if hasattr(opts, key):
                setattr(opts, key, value)
            else:
                raise TypeError(f"Unknown option {key!r}")
        return opts

    def _config(self, opts: PipelineOptions) -> Dict[str, Any]:
        from textlens import __version__

        r = self._router
        return {
            "version": __version__,
            "profile": r.profile.name,
            "model": r.explicit_model,
            "backend": r.backend,
            "device": r.device,
            "fallback": r.fallback_override,
            "fallback_models": r.fallback_models,
            "min_confidence": r.min_confidence,
            "dpi": r.dpi,
            "backend_options": {k: v for k, v in r.backend_options.items() if k != "api_key"},
            "ocr": opts.ocr,
            "pages": opts.pages,
            "task": opts.task,
            "prompt": opts.prompt,
            "detect_tables": opts.detect_tables,
            "ocr_images": opts.ocr_images,
            "reuse_ocr_layer": opts.reuse_ocr_layer,
        }

    # ── core entry points ────────────────────────────────────────────────
    def __call__(self, source: Any, **options: Any) -> Result:
        """OCR / extract a document or image and return a :class:`Result`.

        ``options`` override the constructor defaults for this call:
        ``pages``, ``ocr``, ``task``, ``prompt``, ``max_new_tokens``,
        ``detect_tables``, ``ocr_images``, ``reuse_ocr_layer``, ``password``.
        """
        from textlens.core.result import SOURCE_EMPTY  # noqa: F401
        from textlens.inputs.source import open_source
        from textlens.runtime.cache import ResultCache, config_hash, shared_cache

        use_cache = options.pop("cache", None)
        opts = self._options(options)
        t0 = time.perf_counter()
        src = open_source(source)
        cfg = self._config(opts)
        cfg_hash = config_hash(cfg)
        cache_mode = use_cache if isinstance(use_cache, str) else self._cache_mode
        cache: Optional[ResultCache] = None
        if use_cache is not False and (cache_mode or "").lower() != "off":
            cache = shared_cache(cache_mode)
            if cache.mode != "off":
                hit = cache.get(ResultCache.key(src.sha256, cfg_hash))
                if hit is not None:
                    for p in hit.pages:
                        p.provenance.cached = True
                    return hit
        pages = list(Pipeline(self._router, opts).run(src))
        result = self._assemble(src, pages, opts, cfg, cfg_hash, t0)
        if cache is not None and cache.mode != "off":
            cache.put(ResultCache.key(src.sha256, cfg_hash), result)
        return result

    process = __call__

    def stream(self, source: Any, **options: Any) -> Iterator[Page]:
        """Yield pages as soon as each is processed (constant memory).

        >>> for page in ocr.stream("big.pdf"):
        ...     print(page.number, page.provenance.source, len(page.text))
        """
        from textlens.inputs.source import open_source

        opts = self._options(options)
        yield from Pipeline(self._router, opts).run(open_source(source))

    def read(self, source: Any, prompt: Optional[str] = None, **kwargs: Any) -> str:
        """0.x-compatible: return plain text (``--- Page N ---`` between PDF pages)."""
        dpi = kwargs.pop("dpi", None)
        if dpi is not None and self._router.dpi is None:
            self._router.dpi = int(dpi)
        passthrough = {k: kwargs.pop(k) for k in list(kwargs) if k in ("page", "pages", "max_new_tokens", "task", "ocr")}
        if prompt is not None and prompt != "Text Recognition:":
            passthrough["prompt"] = prompt
        result = self(source, **passthrough)
        texts = [p.markdown if (p.markdown and p.provenance.source == "vlm") else p.text for p in result.pages]
        if len(texts) <= 1:
            return texts[0] if texts else ""
        return "\n\n".join(f"--- Page {p.number} ---\n{t}" for p, t in zip(result.pages, texts))

    def document(self, source: Any, **options: Any) -> Result:
        """Structure-first processing (Markdown task: headings, tables, formulas)."""
        options.setdefault("task", "markdown")
        return self(source, **options)

    def batch(
        self,
        sources: Union[str, Iterable[Any]],
        workers: Optional[int] = None,
        on_result: Optional[Callable[[Any, Union[Result, Exception]], None]] = None,
        recursive: bool = True,
        **options: Any,
    ) -> List[Union[Result, Exception]]:
        """Process many inputs; failures are returned, not raised.

        ``sources`` may be a folder, a glob pattern or an iterable of inputs.
        Workers share one loaded model per engine (no per-thread copies).
        """
        items = list(_expand_sources(sources, recursive))
        n_workers = workers or (1 if not items else min(4, len(items)))

        def run(item: Any) -> Union[Result, Exception]:
            try:
                res: Union[Result, Exception] = self(item, **dict(options))
            except Exception as exc:  # noqa: BLE001 - reported per item
                res = exc
            if on_result is not None:
                try:
                    on_result(item, res)
                except Exception:
                    logger.debug("on_result callback failed", exc_info=True)
            return res

        if n_workers <= 1:
            return [run(i) for i in items]
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="textlens-batch") as pool:
            return list(pool.map(run, items))

    # ── introspection ────────────────────────────────────────────────────
    def inspect(self, source: Any, **kwargs: Any) -> Any:
        """Cheap analysis without OCR: PDF classification or image signals."""
        from textlens.inputs.source import open_source

        src = open_source(source)
        if src.is_pdf:
            from textlens.documents.pdf.inspector import inspect_pdf

            return inspect_pdf(src, password=kwargs.get("password") or self._defaults.password, pages=kwargs.get("pages"), sample=kwargs.get("sample"))
        if src.is_image:
            from textlens.core.difficulty import analyze_image
            from textlens.inputs.images import load_image

            img = src.image if src.image is not None else load_image(src.open_binary())
            return analyze_image(img)
        return {"kind": src.kind, "mime": src.mime, "name": src.name}

    def explain(self, source: Any = None, **kwargs: Any) -> List[RoutingDecision]:
        """Dry-run routing: which model would handle each OCR page, and why."""
        if source is None:
            return [self._router.route(PageSignals(task=self._defaults.task))]
        from textlens.inputs.source import open_source

        src = open_source(source)
        if src.is_pdf:
            insp = self.inspect(src, **kwargs)
            decisions = []
            for p in insp.pages:
                if p.needs_ocr or self._defaults.ocr == "force":
                    d = self._router.route(PageSignals(kind=p.kind, reasons=p.reasons, likely_table=p.likely_table, likely_formulas=p.likely_formulas, has_ocr_layer=p.has_ocr_layer, task=self._defaults.task))
                    d.reasons.insert(0, f"page {p.number}: {p.kind}")
                    decisions.append(d)
            return decisions
        sig = self.inspect(src)
        return [self._router.route(PageSignals(kind="image", likely_table=sig.likely_table, difficulty=sig.difficulty, task=self._defaults.task))]

    def extract(self, source: Any, schema: Any, strategy: str = "auto", **options: Any) -> Any:
        """Structured extraction: ``ocr.extract("invoice.pdf", schema={"total": "float"})``."""
        from textlens.core.extract import Extraction, build_prompt, coerce, extract_heuristic, normalize_schema

        fields = normalize_schema(schema)
        result = self(source, **options)
        if strategy in ("auto", "vlm"):
            answer = self._vlm_extract(source, build_prompt(fields), strict=strategy == "vlm")
            if answer is not None:
                data = {f.name: (coerce(str(answer.get(f.name)), f.type) if answer.get(f.name) is not None else None) for f in fields}
                ext = Extraction(data=data, fields={k: {"value": v, "method": "vlm"} for k, v in data.items()}, method="vlm", result=result)
                result.extraction = ext.to_dict()
                return ext
        ext = extract_heuristic(result, fields)
        result.extraction = ext.to_dict()
        return ext

    def _vlm_extract(self, source: Any, prompt: str, strict: bool) -> Optional[Dict[str, Any]]:
        from textlens.backends.base import RecognizeOptions
        from textlens.backends.loader import get_backend
        from textlens.core.extract import parse_json_answer
        from textlens.inputs.source import open_source
        from textlens.models.specs import TASK_EXTRACTION

        try:
            decision = self._router.route(PageSignals(task=TASK_EXTRACTION))
            spec = self._router.spec_for(decision.model)
            if not spec.is_generative:
                if strict:
                    from textlens.errors import RoutingError

                    raise RoutingError("No generative model is available for VLM extraction.", hint="Install one (textlens models install glm-ocr) or configure an endpoint.")
                return None
            src = open_source(source)
            if src.is_pdf:
                from textlens.documents.pdf.pdfium import PDFIUM_LOCK, open_pdf, render_page

                with open_pdf(src.data if src.data is not None else src.path) as doc, PDFIUM_LOCK:
                    page = doc[0]
                    img, _ = render_page(page, decision.dpi)
                    page.close()
            else:
                from textlens.inputs.images import load_image

                img = src.image if src.image is not None else load_image(src.open_binary())
            device, opts = self._router.options_for(spec.id)
            out = get_backend(spec, device, **opts).recognize([img], RecognizeOptions(task=TASK_EXTRACTION, prompt=prompt))[0]
            return parse_json_answer(out.text)
        except Exception as exc:
            if strict:
                raise
            logger.info("VLM extraction unavailable (%s); using heuristics", exc)
            return None

    def load(self, source: Any, **options: Any) -> "Any":
        """Open a lazy :class:`~textlens.documents.document.Document`."""
        from textlens.documents.document import Document

        return Document(source, ocr=self, **options)

    # ── lifecycle ────────────────────────────────────────────────────────
    def warmup(self) -> "OCR":
        """Load the default model now (useful before serving traffic)."""
        from textlens.backends.loader import get_backend

        d = self._router.route(PageSignals(task=self._defaults.task))
        device, opts = self._router.options_for(d.model)
        get_backend(self._router.spec_for(d.model), device, **opts).warmup()
        return self

    def unload(self) -> None:
        """Free the models this process loaded."""
        from textlens.backends.loader import get_pool

        get_pool().unload()

    @classmethod
    def ensure(cls, model_id: str) -> None:
        """Download a model without loading it (container builds, setup scripts)."""
        from textlens.models import artifacts
        from textlens.models.specs import get_spec

        artifacts.install(get_spec(model_id))

    def __repr__(self) -> str:
        pinned = self._router.explicit_model or "auto"
        return f"<OCR profile={self.profile!r} model={pinned!r} device={self._device or 'auto'!r}>"

    # ── result assembly ──────────────────────────────────────────────────
    def _assemble(self, src: Any, pages: List[Page], opts: PipelineOptions, cfg: Dict[str, Any], cfg_hash: str, t0: float) -> Result:
        from textlens import __version__
        from textlens.documents.layout import mark_repeated_furniture

        if src.is_pdf and len(pages) >= 3:
            mark_repeated_furniture([p.blocks for p in pages], [p.height for p in pages])
        models = sorted({p.provenance.model for p in pages if p.provenance.model})
        backends = sorted({p.provenance.backend for p in pages if p.provenance.backend})
        counts: Dict[str, int] = {}
        for p in pages:
            counts[p.provenance.source] = counts.get(p.provenance.source, 0) + 1
        fallbacks = sum(1 for p in pages if len(p.provenance.attempts) > 1)
        metadata: Dict[str, Any] = {
            "source_name": src.name,
            "mime": src.mime,
            "kind": src.kind,
            "size_bytes": src.size,
            "page_count": len(pages),
        }
        if src.is_pdf:
            try:
                from textlens.documents.pdf.pdfium import PDFIUM_LOCK, open_pdf

                with open_pdf(src.data if src.data is not None else src.path, password=opts.password) as doc, PDFIUM_LOCK:
                    metadata["document_page_count"] = len(doc)
                    metadata.update({k.lower(): v for k, v in (doc.get_metadata_dict() or {}).items() if v})
            except Exception:
                pass
        provenance = RunProvenance(
            textlens_version=__version__,
            profile=self._router.profile.name,
            document_hash=src.sha256,
            config_hash=cfg_hash,
            source=src.name,
            models=models,
            backends=backends,
            device=next((p.provenance.device for p in pages if p.provenance.device), None),
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            routing={"pages_by_source": counts, "pages_with_fallback": fallbacks, "requested_profile": self._router.requested_profile},
        )
        return Result(pages=pages, metadata=metadata, provenance=provenance)


def _expand_sources(sources: Any, recursive: bool) -> Iterator[Any]:
    from pathlib import Path

    from textlens.inputs.source import SUPPORTED_EXTENSIONS

    if isinstance(sources, (str, Path)):
        p = Path(sources)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            yield from sorted(f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
            return
        if any(ch in str(sources) for ch in "*?["):
            yield from sorted(Path().glob(str(sources)))
            return
        yield sources
        return
    yield from sources
