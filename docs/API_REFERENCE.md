# TextLens API reference

This reference describes the public APIs in TextLens 0.2.0. Private methods beginning with `_` are implementation details and may change without notice.

## Primary OCR interfaces

### `OCR(model=None, device=None, auto_download=True, **kwargs)`

The registry-based OCR client. `model` is one of `glm-ocr`, `lighton-ocr`, `hunyuan-ocr`, or `smolvlm`; omitted means the registered default, `glm-ocr`. `device` may be `cuda` or `cpu`; omit it to let the backend choose. Missing registered models download automatically unless `auto_download=False`. Extra keyword arguments go to the backend constructor.

| Member | Description |
| --- | --- |
| `read(source, prompt="Text Recognition:", **kwargs) -> str` | Runs OCR. `source` may be an image/PDF path, URL, Pillow image, bytes, or `BytesIO`. Backend options commonly include `dpi`, `page`, and `max_new_tokens`. Multi-page PDFs are returned as text separated by page headings. |
| `model_id` | Read-only canonical model ID. |
| `model_name` | Read-only human-readable model name. |
| `device` | Requested device value. |
| `is_loaded` | `True` after the backend has loaded. |
| `OCR.ensure(model_id)` | Class method that downloads a registered model if it is not already cached. |

Raises `UnknownModelError` for an unknown model and `ModelNotInstalledError` when automatic download is disabled.

### `TextLens(model_id="zai-org/GLM-OCR", device=None, torch_dtype=None, auto_load=True, auto_fix_dependencies=True, show_progress=True)`

The original GLM-OCR SDK. It is still supported and offers specialised extraction helpers. It chooses CUDA when available, otherwise CPU. `auto_load=True` loads the model during construction.

| Method/property | Description |
| --- | --- |
| `load() -> None` | Loads the processor and model weights. Called automatically by `read()` if needed. |
| `is_loaded` | Whether weights are currently in memory. |
| `hardware` | Current `HardwareInfo` snapshot. |
| `is_cuda() -> bool` | Whether this engine is actively using available CUDA. |
| `switch_device(target_device) -> str` | Moves loaded weights to `cuda`/`gpu` or `cpu`; raises if not loaded or CUDA is unavailable. |
| `read(image_source, prompt="Text Recognition:", max_new_tokens=512, temperature=0.7, top_p=0.95) -> str` | Reads one image, URL, local path, or Pillow image. |
| `read_pdf(pdf_source, prompt="Text Recognition:", scale=2.0, max_pages=None, max_new_tokens=512) -> list[dict]` | Renders and OCRs PDF pages. Each result has `page`, `total_pages`, and `text`. |
| `extract_table(image_source) -> str` | Uses a Markdown-table extraction prompt. |
| `extract_formula(image_source) -> str` | Uses a LaTeX-formula extraction prompt. |
| `extract_json(image_source, schema=None) -> str` | Uses a structured JSON prompt; `schema` is appended as guidance. Validate the returned text before treating it as parsed JSON. |
| `batch_read(sources, prompt="Text Recognition:") -> list[str]` | Sequentially OCRs image/URL sources. For folder-scale work, prefer `BatchOCR`. |
| `serve(host="0.0.0.0", port=8000, reload=False) -> None` | Starts the REST server using this loaded engine. |

## Batch OCR

### `BatchOCR(...)`

Constructor parameters: `model="glm-ocr"`, `workers=4`, `output_format="json"`, `output_dir="./batch_output"`, `retries=2`, `dpi=200`, `device=None`, `enable_dashboard=True`, `dashboard_port=8765`, `dashboard_host="127.0.0.1"`, `recursive=True`, `max_new_tokens=2048`, and `queue_backend=None`.

| Method | Description |
| --- | --- |
| `run(source, on_file_complete=None, on_file_failed=None) -> list[BatchTask]` | Blocks until the directory or explicit list finishes/cancels. Completion and permanent-failure callbacks receive the associated task. |
| `pause()` / `resume()` | Stops new work / lets queued work continue. In-flight work completes. |
| `cancel()` | Signals workers to exit after current work. |
| `retry_failed() -> int` | Requeues permanently failed tasks and returns the count. |
| `reconfigure(workers=None, output_format=None, retries=None)` | Updates future batch settings; increasing workers starts additional threads. |
| `get_metrics() -> JobMetrics` | Current metrics, including status, progress, speed, ETA, CPU/RAM/VRAM fields. |
| `get_tasks() -> list[BatchTask]` | Task snapshot. |
| `get_logs(last_n=100) -> list[str]` | Recent engine log lines. |

`BatchTask.to_dict()` and `JobMetrics.to_dict()` produce JSON-safe monitoring/export dictionaries. Task statuses are `QUEUED`, `PROCESSING`, `RETRYING`, `COMPLETED`, and `FAILED`; batch statuses are `IDLE`, `RUNNING`, `PAUSED`, `CANCELLED`, `COMPLETED`, and `FAILED`.

`BatchJobConfig` is the configuration dataclass used internally by `BatchOCR`. Its fields mirror the constructor settings and additionally expose `input_source` and an optional `prompt`. Treat it as configuration data rather than a job controller.

### Batch execution lifecycle

`BatchOCR.run(source)` first discovers supported files (PDF, PNG, JPG/JPEG, WEBP, BMP, TIFF/TIF), then queues a `BatchTask` for every input. Worker threads each create an OCR engine, process and export each task, retry failed tasks up to `retries`, and finally write `batch_manifest.json` in `output_dir`. `workers=1` is the safest starting point on a single GPU; raise it only after confirming the selected model fits comfortably in VRAM. The controller methods `pause()`, `resume()`, `cancel()`, `retry_failed()`, `metrics()`, and `logs()` are safe to use while a job is running.

### Batch extension points

`BaseBatchQueue` defines `enqueue`, `dequeue`, `task_done`, `requeue_failed`, `get_task`, `get_all_tasks`, `size`, and `clear`. `MemoryBatchQueue` is the provided thread-safe in-memory implementation. Implement every method to use a custom queue backend.

`StructuredExporter(output_dir, default_format="json")` exposes:

| Method | Description |
| --- | --- |
| `export_task(task, model_id, output_format=None) -> Path` | Writes a completed task as JSON, Markdown, CSV, or plain text. |
| `export_summary_manifest(tasks, model_id, elapsed_sec) -> Path` | Writes `batch_manifest.json`, always using JSON. |

## Models and cache

| API | Description |
| --- | --- |
| `ModelRegistry.all()` | Returns registered `ModelMetadata` in catalog order. |
| `ModelRegistry.supported_ids()` | Returns canonical IDs. |
| `ModelRegistry.get(model_id)` | Resolves a canonical ID or normalized display name; raises `UnknownModelError` if unavailable. |
| `ModelRegistry.is_registered(model_id)` | Checks an ID. Prefer `get()` when aliases/display names matter. |
| `ModelRegistry.default()` | Returns the default metadata. |
| `ModelManager.models()` | Prints and returns the catalog. |
| `ModelManager.download(model_id)` / `remove(model_id)` | Downloads/removes a registered model. |
| `ModelManager.info(model_id) -> ModelMetadata` | Prints and returns detailed metadata/cache status. |
| `ModelManager.is_installed(model_id) -> bool` | Validates and checks the cache. |
| `discover_models(search="ocr", use_case=None, limit=12, compatible_only=False, profile=None) -> list[DiscoveredModel]` | Queries live Hugging Face OCR/VLM candidates and adds parameter/VRAM guidance using a detected or supplied `HardwareProfile`. Official catalog entries retain their registered minimum VRAM; third-party figures are estimates. Install `textlens-srevarshan[catalog]`. These are suggestions, not supported TextLens backends. |
| `ModelCache.model_path(model_id)` | Local cache path (`~/.cache/textlens/models/<model-id>` by default). |
| `ModelCache.is_installed`, `disk_usage_gb`, `ensure_directory`, `list_installed`, `remove` | Cache inspection and management utilities. |
| `ModelDownloader.download(model_id, force=False) -> Path` | Lower-level Hugging Face download. |

`ModelMetadata` is an immutable descriptor. `use_cases_str(sep="\n  ")` formats its use cases and `short_label()` returns a compact label.

## Hardware and dependencies

| Function | Description |
| --- | --- |
| `inspect_hardware() -> HardwareProfile` | New hardware inspector: OS, CPU/RAM, GPU/VRAM and CUDA snapshot. |
| `HardwareDoctor().run() -> DoctorReport` | Applies deterministic VRAM/device recommendations to each model. |
| `HardwareDoctor().print_report(report)` | Prints the diagnostic report. |
| `detect_system_cuda() -> SystemCUDADetails` | Uses `nvidia-smi` to inspect an NVIDIA driver. |
| `get_pytorch_cuda_install_cmd(cuda_version) -> str` | Returns TextLens's suggested PyTorch install command. Check the official PyTorch selector before using it because wheel availability changes. |
| `is_cuda_available() -> bool` | True only if this Python's PyTorch build can use CUDA. |
| `get_hardware_info() -> HardwareInfo` | Legacy hardware snapshot. `HardwareInfo.to_dict()` is JSON-safe. |
| `print_hardware_status()` | Human-readable legacy CUDA diagnostic. |
| `check_dependencies() -> DependencyReport` | Reports installed/missing runtime modules and an install command. |
| `ensure_dependencies(auto_install=False, verbose=True) -> bool` | Checks dependencies; with `auto_install=True`, attempts to install missing packages. |

## Server and CLI

`create_app(engine=None) -> FastAPI` builds the app; inject a preloaded `TextLens` engine to avoid startup loading. `serve(host="127.0.0.1", port=8000, reload=False, engine=None)` launches Uvicorn.

The CLI entry point is `textlens`. Run `textlens --help` for current syntax. Public commands are `models`, `model install/remove/info`, `doctor`, `read`, `batch`, and `serve`.

## Test matrix

| Feature group | Automated coverage | Command |
| --- | --- | --- |
| Registry, manager, downloader errors | `tests/test_registry.py`, `tests/test_manager.py` | `python -m pytest tests/test_registry.py tests/test_manager.py -q` |
| Hardware and doctor | `tests/test_hardware.py`, `tests/test_doctor.py` | `python -m pytest tests/test_hardware.py tests/test_doctor.py -q` |
| Legacy SDK | `tests/test_sdk.py` | `python -m pytest tests/test_sdk.py -q` |
| CLI | `tests/test_cli.py` | `python -m pytest tests/test_cli.py -q` |
| REST app | `tests/test_server.py` | `python -m pytest tests/test_server.py -q` |
| Batch queue, exports, controls, dashboard | `tests/test_batch.py` | `python -m pytest tests/test_batch.py -q` |
| Whole offline suite | all test files | `python scripts/run_feature_checks.py` |
| Optional real VLM inference | image supplied by developer | `python scripts/run_feature_checks.py --image PATH --model glm-ocr --device cuda` |

The suite intentionally does not download models or run a real VLM. The optional real check does, so it requires a downloaded model, sufficient disk/VRAM, and an internet connection if the model is not cached.

## Lower-level utilities and backend contracts

These APIs are useful when extending TextLens, but ordinary applications should use `OCR`, `TextLens`, and `BatchOCR` instead.

| API | Description |
| --- | --- |
| `render_pdf_to_images(pdf_source, dpi=200, page_selection=None)` | Converts a local PDF, PDF bytes, or `BytesIO` to Pillow images for a backend. |
| `load_input_images(source, dpi=200, page=None)` | Normalises supported image/PDF inputs into image objects. |
| `ProgressTracker(total_steps=100, desc="Processing")` | Lightweight terminal progress helper: call `update(step, message=None)` and `complete(message="Done!")`. `complete()` returns elapsed seconds. |
| `print_step(step_num, total, title)` | Prints a formatted progress step. |
| `BaseOCRModel(device=None)` | Abstract backend contract. Implement `load()`, `predict(source, prompt, **kwargs)`, `download()`, `is_installed()`, `metadata()`, and `device_requirements()`. Its `is_loaded` and `device` properties report runtime state. |
| `GLMOCRBackend`, `LightOnOCRBackend`, `HunyuanOCRBackend`, `SmolVLMBackend` | Model-specific `BaseOCRModel` implementations selected by `OCR`. Their public `load`, `predict`, `download`, `is_installed`, `metadata`, and `device_requirements` methods follow the base contract. |

Custom backends should keep `predict()` output as a string and should clearly reject unsupported device/model combinations. The public exception hierarchy is `TextLensError`, `UnknownModelError`, `ModelNotInstalledError`, `DownloadError`, and `HardwareInspectionError`.
