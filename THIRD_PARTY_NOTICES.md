# Third-party notices

TextLens is MIT-licensed. It builds on, or was informed by, the projects
below. Model **weights** are never bundled in the TextLens package; they are
downloaded at runtime under their own licenses (see `textlens models info <id>`).

## Code and design references

### Firecrawl pdf-inspector — MIT License
Copyright (c) 2026 Firecrawl — https://github.com/firecrawl/pdf-inspector

TextLens's PDF inspector and text-quality checks follow pdf-inspector's
design: per-page OCR reason codes (`scanned`, `no_text`, `vector_text`,
`invisible_text_layer`, `suspected_garbled_text`), sampling strategy,
selective OCR routing, and several calibrated text-quality thresholds
(replacement-character density, dollar-as-space, substitution-cipher letter
statistics, CID/C1 garbage). TextLens's implementation is an independent
Python implementation built on pypdfium2; no Rust source was copied.

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions: The above copyright
> notice and this permission notice shall be included in all copies or
> substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
> WITHOUT WARRANTY OF ANY KIND.

### The Neural Maze — Production OCR Course — Apache License 2.0
https://github.com/neural-maze/production-ocr-course

Architecture reference for the collector-pattern dynamic batcher,
queue-decoupled workers, backpressure, workspace-confined MCP tools and the
production documentation structure. No code was copied.

## Runtime dependencies (not bundled)

| Package | License |
|---|---|
| pypdfium2 / PDFium | Apache-2.0 / BSD-3-Clause |
| ONNX Runtime | MIT |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |

## Models downloaded on demand

| Model | License | Notes |
|---|---|---|
| PP-OCRv6 Small (ONNX export by GreatV/oar-ocr v0.7.0) | Apache-2.0 | default engine, SHA-256 pinned |
| GLM-OCR (zai-org) | MIT | |
| LightOnOCR-2-1B | Apache-2.0 | |
| SmolVLM-256M-Instruct | Apache-2.0 | |
| HunyuanOCR (Tencent) | tencent-hunyuan-community | restricted — review before use |
| PaddleOCR-VL | Apache-2.0 | served via OpenAI-compatible endpoint |
| DeepSeek-OCR / DeepSeek-OCR-2 | MIT / Apache-2.0 | served via OpenAI-compatible endpoint |
| open-image-models / fast-plate-ocr (ANPR) | MIT | optional `anpr` extra |

Licenses were read from each model's Hugging Face card metadata on
2026-09-29. Always check the current model card before commercial use.
