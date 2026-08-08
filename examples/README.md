# TextLens examples

Each script demonstrates one feature and is intended to be run from the repository root after installation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

| Script | Purpose | Downloads/loads a model? |
| --- | --- | --- |
| `00_environment_check.py` | Inspect Python, CUDA, GPU, and dependency status | No |
| `01_model_catalog.py` | List models and inspect a selected model | No |
| `02_basic_ocr.py IMAGE` | OCR one local image with the modern `OCR` API | Yes |
| `03_custom_prompt.py IMAGE` | OCR with an instruction prompt | Yes |
| `04_pdf_ocr.py PDF` | Extract text page-by-page with the legacy SDK | Yes |
| `05_structured_extraction.py IMAGE` | Table, formula, and JSON helpers | Yes |
| `06_device_switching.py IMAGE` | Load GLM-OCR then switch CPU/CUDA | Yes |
| `07_batch_folder.py FOLDER` | BatchOCR folder processing and exports | Yes |
| `08_batch_callbacks.py FOLDER` | Batch callbacks and result filtering | Yes |
| `09_rest_server.py` | Start the local REST API | Yes |
| `10_rest_client.py IMAGE` | Submit a file to a running REST API | No (server does OCR) |

Start with `00_environment_check.py`. For a CUDA-enabled machine, `CUDA available: True` must appear before expecting GPU acceleration. The first model-based run can download several GB of model weights; subsequent runs use the local cache.

For a safe first model run, use the smaller model:

```powershell
python examples\02_basic_ocr.py .\test-image-ocr.png --model smolvlm --device cuda
```
