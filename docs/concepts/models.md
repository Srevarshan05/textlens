# Models

TextLens does not compete on model count. The registry holds a small set of
engines that cover different hardware and tasks, each described by a
`ModelSpec` the router can reason about.

```bash
textlens models list            # catalog + install status
textlens models search table
textlens models info glm-ocr    # model card
textlens models install glm-ocr
textlens models verify ppocrv6-small   # re-check SHA-256 of installed files
textlens models remove glm-ocr
textlens models path glm-ocr
```

## The catalog

| Model | Backend | Runs on | Tasks | License | Status |
|---|---|---|---|---|---|
| **ppocrv6-small** (default) | ONNX Runtime | any CPU, Pi, Jetson; GPU optional | text lines + boxes + confidence | Apache-2.0 | local |
| glm-ocr | Transformers | GPU ≥ 4 GB (6 GB rec.), CPU slow | text, tables, formulas, Markdown | MIT | local |
| lighton-ocr | Transformers | GPU ≥ 6 GB | text, tables, formulas, Markdown | Apache-2.0 | local |
| hunyuan-ocr | Transformers | GPU ≥ 6 GB | layouts, charts, extraction | Tencent community (restricted) | local |
| smolvlm | Transformers | CPU or small GPU | rough text | Apache-2.0 | local |
| paddleocr-vl | OpenAI-compatible | GPU server (vLLM…) | text, tables, formulas, charts; 100+ languages | Apache-2.0 | served |
| deepseek-ocr-2 | OpenAI-compatible | GPU server ≥ 10 GB | dense long documents | Apache-2.0 | served |
| deepseek-ocr | OpenAI-compatible | GPU server ≥ 10 GB | documents | MIT | served |
| mineru2.5 | — | — | document parsing | **AGPL-3.0** | catalog only |
| surya | — | — | OCR, layout, reading order | GPL-3.0 code, restricted weights | catalog only |

Licenses come from each model's Hugging Face card (checked 2026-09-29). Always
check the current card before commercial use; `textlens models info`
prints license notes and known limitations.

- **local** — TextLens has an adapter and downloads weights.
- **served** — run it in an inference server (vLLM, SGLang…) and point
  TextLens at it: `OCR(model="paddleocr-vl", endpoint="http://host:8000/v1")`.
- **catalog only** — documented because people ask; not bundled, usually for
  license reasons. Serve it yourself behind an OpenAI-compatible API if its
  license suits you.

## What a spec contains

```python
from textlens.models.specs import get_spec
spec = get_spec("glm-ocr")
spec.tasks, spec.languages, spec.boxes, spec.confidence, spec.markdown
spec.cpu, spec.cpu_practical, spec.cuda, spec.edge
spec.min_vram_gb, spec.recommended_vram_gb, spec.min_ram_gb, spec.quantization
spec.requires, spec.extra             # runtime modules and the pip extra that installs them
spec.hf_repo_id, spec.revision, spec.artifacts   # weights (artifacts carry SHA-256)
spec.license, spec.commercial_use, spec.license_notes, spec.limitations
spec.quality_tier, spec.speed_tier    # routing heuristics, not benchmarks
```

## Downloads and integrity

- Pinned URL artifacts (PP-OCRv6) are streamed to a temp file, SHA-256
  verified, then atomically renamed. A corrupted or tampered file is never
  installed.
- Hugging Face models use `snapshot_download`, pinned to `spec.revision` when
  the spec sets one.
- `TEXTLENS_OFFLINE=1` turns every download into an error with instructions.
- Small models (≤ 100 MB) download on first use; large ones only when you
  pin them or run `textlens models install`.

## Adding your own engine

1. Implement an adapter:

```python
from textlens.backends.base import OCRBackend, PageOCR
from textlens.documents.layout import Item
from textlens.core.result import BBox

class MyEngine(OCRBackend):
    thread_safe = True                      # may run concurrently
    backend_name = "my-runtime"

    def _load(self):
        import my_runtime                   # heavy imports only here
        self.model = my_runtime.load(self.options.get("weights"))
        self.device = "cpu"

    def _recognize(self, images, options):
        out = []
        for img in images:
            lines = self.model.read(img)    # [(text, (x0, y0, x1, y1), conf)]
            items = [Item(t, BBox(*b), c) for t, b, c in lines]
            out.append(PageOCR(width=img.width, height=img.height, items=items))
        return out
```

Return `items` (positioned lines) for detection-style engines, or
`markdown=` for generative ones — the pipeline builds blocks, tables and
reading order either way.

2. Register a spec (in code, or via the `textlens.models` entry-point group):

```python
from textlens import ModelSpec, register_spec

register_spec(ModelSpec(
    id="my-engine", display_name="My Engine", family="custom", provider="me",
    description="…", backend="onnx", adapter="my_package.engine:MyEngine",
    tasks=frozenset({"text"}), boxes=True, confidence=True, cpu=True, cpu_practical=True,
    requires=("my_runtime",), license="apache-2.0", commercial_use="yes",
))
```

```toml
# pyproject.toml of your plugin package
[project.entry-points."textlens.models"]
my-engine = "my_package.engine:specs"   # a ModelSpec, a list, or a callable returning them
```

The router, doctor, CLI, server and benchmark pick it up automatically.

## Discovering new models

`textlens discover <name>` searches the Hugging Face Hub and rates results
against your GPU. Discovered repositories are research suggestions, not
supported backends, until an adapter and spec exist.
