# API reference (TextLens 2.0)

Public names are importable from `textlens` (lazily — importing `textlens`
loads no ML runtime). Anything prefixed `_` is private.

## `OCR`

```python
OCR(model=None, device=None, auto_download=True, *, profile=None, backend=None,
    endpoint=None, api_key=None, fallback=None, fallback_models=None,
    min_confidence=None, latency_budget_ms=None, memory_budget_gb=None,
    quality=None, ocr="auto", dpi=None, task="text", cache=None,
    allowed_licenses=None, commercial_only=False, detect_tables=True,
    ocr_images=False, reuse_ocr_layer=None, password=None, **backend_options)
```

| Parameter | Meaning |
|---|---|
| `model` | pin a registered model id/alias, or any served model with `backend="openai"` |
| `device` | `cpu`, `cuda`, `cuda:N`, `tensorrt`, `coreml`, `dml` |
| `auto_download` | download missing weights on first use (large models only when pinned) |
| `profile` | `auto`, `edge`, `fast`, `balanced`, `accurate`, `document`, `server` |
| `backend`, `endpoint`, `api_key` | remote OpenAI-compatible inference |
| `fallback`, `fallback_models`, `min_confidence` | confidence-validated fallback chain |
| `latency_budget_ms`, `memory_budget_gb` | routing budgets |
| `quality` | `"high"` / `"low"` nudges `auto` |
| `ocr` | `auto` (selective), `force`, `off` |
| `dpi` | render resolution for OCR'd PDF pages |
| `task` | `text`, `markdown`, `table`, `formula` (generative models) |
| `cache` | `off`, `memory`, `disk` |
| `allowed_licenses`, `commercial_only` | license policy for routing |
| `detect_tables`, `ocr_images`, `reuse_ocr_layer`, `password` | PDF behaviour |
| `**backend_options` | adapter options, e.g. `threads`, `low_memory`, `det_limit`, `concurrency` |

| Method / property | Returns |
|---|---|
| `ocr(source, **overrides)` / `ocr.process(...)` | `Result`. Overrides: `pages`, `ocr`, `task`, `prompt`, `max_new_tokens`, `detect_tables`, `ocr_images`, `reuse_ocr_layer`, `password`, `cache`, `events` |
| `ocr.read(source, prompt=None, **kw)` | `str` (0.x compatible; `--- Page N ---` between pages) |
| `ocr.document(source, **kw)` | `Result` with the Markdown task |
| `ocr.stream(source, **kw)` | iterator of `Page` (constant memory) |
| `ocr.batch(sources, workers=None, on_result=None, recursive=True, **kw)` | list of `Result` or exception per input (folder, glob or iterable) |
| `ocr.inspect(source, pages=None, sample=None)` | `DocumentInspection` (PDF) or `ImageSignals` (image) |
| `ocr.explain(source=None)` | list of `RoutingDecision` (dry run, no OCR) |
| `ocr.extract(source, schema, strategy="auto")` | `Extraction(data, fields, method, result)` |
| `ocr.load(source)` | lazy `Document` |
| `ocr.warmup()`, `ocr.unload()` | load / free models |
| `OCR.ensure(model_id)` | download without loading |
| `profile`, `router`, `model_id`, `model_name`, `device`, `is_loaded` | properties |

`source` may be a path, `http(s)` URL, `bytes`, a binary file object, a PIL
image or a numpy array; types are detected from content.

## `Result` and friends — `textlens.core.result`

`Result(pages, metadata, provenance, extraction)` with properties `text`,
`blocks`, `lines`, `words`, `tables`, `formulas`, `layout`, `confidence`,
`page_count` and methods `to_text()`, `to_markdown(page_markers=False,
include_furniture=False)`, `to_html(full_document=True)`, `to_json()`,
`to_dict()`, `to_csv()`, `to_chunks(max_tokens=500, overlap_tokens=50,
strategy="semantic", tokenizer=None, include_furniture=False)`,
`save(directory, formats)`, `pages_needing_review()`, `Result.from_json()`,
`Result.from_dict()`.

`Page(number, width, height, unit, blocks, tables, formulas, provenance,
classification, rotation, markdown)` · `ordered_blocks()`, `text`, `lines`,
`words`, `confidence`, `to_markdown()`.

`Block(type, text, bbox, confidence, lines, page, source, model, backend,
level, order, id, attributes)` · `Line(text, bbox, confidence, words,
polygon, font_size, bold)` · `Word(text, bbox, confidence)` ·
`Table(cells, page, bbox, confidence, source, model, caption, id)` with
`grid()`, `to_markdown()`, `to_csv()`, `to_html()`, `to_records()`,
`Table.from_grid()` · `TableCell(row, col, text, row_span, col_span, bbox,
confidence, is_header)` · `Formula(latex, page, bbox, confidence, source,
model)` · `BBox(x0, y0, x1, y1)` with `width`, `height`, `area`, `center`,
`union`, `intersection`, `iou`, `scale`, `from_points`.

`PageProvenance(page, source, model, model_revision, backend, device, dpi,
confidence, reasons, routing, attempts, timings_ms, warnings, needs_review,
cached)` · `Attempt(model, backend, confidence, accepted, reason,
elapsed_ms)` · `RunProvenance(textlens_version, schema_version, profile,
document_hash, config_hash, source, models, backends, device, created_at,
elapsed_ms, routing)` · `Chunk(text, index, pages, bboxes, section,
block_ids, document_id, metadata)`.

## Documents — `textlens.load(source, ocr=None, **options)`

`Document`: `page_count`, `name`, `kind`, `page(n)`, `iter_pages()`,
`pages()`, `result`, `text()`, `tables()`, `layout()`, `images()`,
`metadata()`, `inspect()`, `to_markdown()`, `to_json()`, `to_html()`,
`to_dict()`, `chunks(**kw)`.

## PDF inspection — `textlens.inspect_pdf(source, pages=None, sample=None, password=None, config=None)`

Returns `DocumentInspection(page_count, pdf_type, confidence, pages,
pages_needing_ocr, reasons_by_page, sampled, metadata, elapsed_ms)`;
each `PageInspection` has `number, width, height, rotation, kind, needs_ocr,
reasons, text_chars, visible_alnum, invisible_chars, unicode_errors,
text_objects, image_objects, path_objects, form_objects, path_segments,
image_coverage, text_coverage, image_boxes, largest_image_px, image_dpi,
horizontal_rules, vertical_rules, math_signals, quality_signals,
has_ocr_layer, likely_table, likely_formulas, elapsed_ms`. Thresholds live in
`InspectionConfig`.

## Routing — `textlens.core.router`

`Router(profile, *, model, backend, device, endpoint, fallback,
fallback_models, min_confidence, latency_budget_ms, memory_budget_gb,
quality, allow_download, allowed_licenses, commercial_only, dpi,
backend_options, system)` → `route(PageSignals) -> RoutingDecision(model,
fallbacks, profile, dpi, device, options, reasons, rejected, scores,
accept_confidence, review_confidence, fallback_enabled)` with `explain()`.
`PROFILES` maps names to `Profile` settings.

## Models — `textlens.models`

- `ModelSpec` (see [Models](concepts/models.md)), `register_spec(spec, replace=False)`,
  `get_spec(id)`, `find_spec(id)`, `all_specs()`, `search_specs(q)`, `default_spec()`
  in `textlens.models.specs`.
- `textlens.models.artifacts`: `install(spec, force=False)`, `is_installed(spec)`,
  `verify(spec)`, `model_dir(spec)`.
- 0.x: `ModelRegistry` (`all`, `get`, `spec`, `specs`, `supported_ids`,
  `default`, `is_registered`), `ModelManager` (`models`, `download`, `info`,
  `remove`, `is_installed`), `ModelCache`, `ModelDownloader`, `HardwareDoctor`
  (`run(deep=False)`, `print_report`), `inspect_hardware()`, `discover_models()`.

## Backends — `textlens.backends`

`OCRBackend(spec, device=None, **options)`: override `_load()` and
`_recognize(images, options) -> list[PageOCR]`; set `thread_safe`,
`backend_name`. Public: `load()`, `unload()`, `recognize(images, options)`,
`warmup()`, `describe()`. `PageOCR(width, height, items, markdown,
confidence, …)`, `RecognizeOptions(task, prompt, max_new_tokens, language,
extra)`. `textlens.backends.loader.get_backend(spec, device, **opts)`,
`get_pool()` (`loaded()`, `unload(model_id=None)`).

## Runtime — `textlens.runtime`

`inspect_system(refresh=False, deep=False) -> SystemInfo(hardware, platform,
runtimes)`; `ResultCache(mode)`; `JobManager(workers, max_queue,
ttl_seconds, retries, store)` with `submit`, `get`, `wait`, `cancel`,
`stats`, `shutdown`; `JobStore` interface; `DynamicBatcher(process,
max_batch, window_ms, max_queue)`; `suggest_batch_size(spec, system)`.

## Evaluation — `textlens.evaluation`

`metrics`: `cer`, `wer`, `ned`, `exact_match`, `text_scores`,
`table_cell_accuracy`, `levenshtein`. `benchmark`: `load_dataset`,
`run_benchmark(dataset, models=None, profiles=None, limit=None, device=None,
save_measurements=True, warmup=True)`. `profiler.profile_run(ocr, source,
pages=None, repeat=1)`. `visualize.render_overlays(source, out_dir, ocr=None,
pages=None, dpi=110)`.

## ANPR — `textlens.ANPR`

`ANPR(profile="balanced", region="generic", detector=None, recognizer=None,
vehicle_detector=False, device=None, min_confidence=0.5, require_valid=False,
max_plates=20)` → `__call__(image) -> ANPRResult(plates, width, height,
timings_ms, profile, frame)` with `best`, `to_dict()`, `annotate(image)`;
`stream(frames)`; `batch(images)`. `PlateRead(plate, confidence, bbox,
raw_text, detection_confidence, recognition_confidence, char_confidences,
polygon, valid, format, corrections, region, vehicle_type, vehicle_bbox,
track_id, detector, recognizer)`. `textlens.anpr`: `REGIONS`,
`load_region()`, `Region`, `PlateFormat`, `PlateValidator`, `normalize()`.

## Server — `textlens.serving`

`create_app(settings=None, ocr=None, auth=None, engine=None)`,
`serve(settings=None, host=None, port=None, **overrides)`,
`ServerSettings.from_env()`. Endpoints: [server guide](deployment/server.md).
MCP: `textlens.serving.mcp.build_server(workspace)`, `run_mcp(...)`.

## Errors — `textlens.errors`

`TextLensError(message, hint=None, **details)` with `code`, `http_status`,
`to_dict()`. Subclasses: `ConfigurationError`; `InputError` →
`InputNotFoundError` (also `FileNotFoundError`), `UnsupportedInputError`
(also `ValueError`), `InputTooLargeError`, `DocumentError`,
`EncryptedDocumentError`; `ModelError` → `UnknownModelError`,
`ModelNotInstalledError`, `DownloadError`, `IntegrityError`, `OfflineError`;
`BackendError` → `BackendUnavailableError`, `BackendLoadError`,
`InferenceError`; `RoutingError`; `JobError` → `JobNotFoundError`,
`CancelledError`; `CapacityError`; `HardwareInspectionError`; `ANPRError`.

## Configuration — `textlens.config`

`get_settings()`, `reload_settings()`, `write_config_file(values)`,
`Settings` fields: `home`, `models_dir`, `results_dir`, `profile`, `device`,
`offline`, `no_persist`, `result_cache`, `log_level`, `log_format`,
`max_pages`, `max_file_mb`, `allow_urls`, `hf_token`. Environment variables
are listed in the module docstring.

## 0.x compatibility

`TextLens` (legacy GLM-OCR SDK), `BatchOCR` and its types, the 0.x hardware
and dependency helpers, and `textlens.server.create_app/serve` remain
available. See the [migration guide](migration.md).
