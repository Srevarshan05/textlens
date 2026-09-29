# Migrating from TextLens 0.x to 2.0

Most 0.x code keeps working. The notable changes:

## Installation

- Python ≥ 3.10 (was 3.9).
- Core dependencies are now `pillow`, `pypdfium2`, `numpy`, `onnxruntime`.
  `requests` is no longer required.
- Document VLMs need the `gpu` extra: `pip install "textlens-ocr[gpu]"` (the
  0.x `inference` extra still works as an alias).
- TextLens never installs packages implicitly. `TextLens(...)` defaults to
  `auto_fix_dependencies=False`, and the server no longer runs `pip` at start-up.

## `OCR`

| 0.x | 2.0 |
|---|---|
| `OCR()` loaded GLM-OCR | `OCR()` routes: PP-OCRv6 on CPU; a VLM for complex pages when one is installed and fits (`balanced`) |
| `OCR(model="glm-ocr").read(x) -> str` | unchanged |
| multi-page `read()` joined with `--- Page N ---` | unchanged |
| models downloaded at `OCR()` construction | downloaded on first use; `OCR.ensure(id)` still pre-downloads |
| — | `OCR()(x) -> Result` with pages, blocks, boxes, confidence, provenance |

To keep 0.x behaviour exactly: `OCR(model="glm-ocr")`.

GLM-OCR decoding is now greedy (temperature 0) so results are reproducible.
Pass `temperature=` through the legacy `TextLens.read()` if you relied on
sampling.

## Model catalog

- New default: `ppocrv6-small`. `ModelRegistry.default()` returns it.
- `ModelRegistry.all()` returns more entries (served and catalog-only models).
- Display names follow model cards (`GLM-OCR`, `LightOnOCR-2`); lookups by
  0.x names and aliases still work.
- Spec-driven doctor recommendations; `glm-ocr` minimum VRAM is 4 GB
  (6 GB recommended), `lighton-ocr` 6 GB (8 recommended), `hunyuan-ocr` 6 GB
  (8 recommended) and no longer claims CPU support.
- `textlens model install|remove|info` still work; prefer `textlens models …`.

## CLI

- `textlens read` → `textlens ocr` (old name kept).
- `textlens models` (bare) still lists models; new sub-commands: `list`,
  `search`, `install`, `info`, `remove`, `verify`, `path`.
- `-v/--version` unchanged; verbosity is `--verbose`.
- `textlens batch` defaults: `--workers 2`, routed model (was `glm-ocr`),
  `--resume` added; the dashboard is unchanged.

## BatchOCR

`BatchOCR(model=None)` routes per page (was `glm-ocr`). Workers share one
engine instead of loading a model each. JSON exports add `confidence` and the
full `result`.

## REST API

The 0.x routes (`/`, `/api/v1/health`, `/api/v1/hardware`, `/api/v1/ocr`,
`/api/v1/ocr/json-payload`) keep their request/response shapes, with one
security change: **`image_url` no longer accepts local file paths** unless
`TEXTLENS_ALLOWED_ROOT` is set, and URLs only when the server allows them.
New endpoints are documented in [the server guide](deployment/server.md).

`textlens.server.create_app(engine=…)` and `serve(host, port)` still work.

## Exceptions

`textlens.models.exceptions` re-exports the classes from `textlens.errors`;
`except UnknownModelError` etc. keep working. New errors carry `.code` and
`.hint`. `InputNotFoundError` is a `FileNotFoundError`, `UnsupportedInputError`
a `ValueError`.

## Cache

Model weights stay in `~/.cache/textlens/models`; installed models remain
installed. New: `TEXTLENS_HOME`, `TEXTLENS_MODELS_DIR` and friends
([installation](getting-started/installation.md)).
