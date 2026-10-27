# PDF subsystem design and Firecrawl pdf-inspector

TextLens's PDF subsystem was designed after studying the implementation (not
just the README) of [Firecrawl pdf-inspector](https://github.com/firecrawl/pdf-inspector)
(MIT, Rust, ~150k lines, commit `ef52f77`, 2026-09-27). This note records what
was adopted, what differs, and why. No Rust source was copied; the Python
implementation is TextLens's own, built on pypdfium2.

## What pdf-inspector does

1. **Detection** (`detector.rs`): parses content streams with lopdf (no
   rendering), sampling up to 8 pages (first, last, evenly spaced). Per page it
   counts text operators, image `Do` operators, path operators, unique
   characters, font decodability (Identity-H without ToUnicode, Type3-only),
   invisible text (render mode 3/7 under a covering image) and template
   images. It classifies the document as TextBased / Scanned / ImageBased /
   Mixed with a confidence, and each page with OCR reason codes.
2. **Text quality** (`text_quality.rs`): U+FFFD density, private-use and C1
   control runs, dollar-as-space (`Word$Word`), substitution-cipher letter
   statistics (English letter-frequency cosine vs sorted-shape cosine), and
   symbol-soup ratios — calibrated against a 380-document corpus.
3. **Extraction**: position-aware text, column detection by projection
   histograms, newspaper vs tabular reading order, three table detectors
   (rectangles, lines, heuristics), Markdown generation.
4. **Selective OCR** (`vision/`): `off | auto | force`; auto routes only
   recommended pages, renders them with PDFium and runs PP-OCRv6 Small on ONNX
   Runtime; per-page provenance (source native/ocr/fused, model identity +
   revision, DPI, confidence, timings, warnings) and a `hosted_recommended`
   flag for pages whose OCR is weak or incomplete.
5. **Lazy runtime**: documents with no routed pages never load PDFium/ORT or
   download models; artifacts are pinned by SHA-256 with an offline mode.

## Adopted in TextLens

| Concept | TextLens |
|---|---|
| per-page reason codes | same vocabulary: `scanned`, `no_text`, `vector_text`, `invisible_text_layer`, `suspected_garbled_text` (+ `sparse_text_over_image`) |
| sampling strategy | `inspect_pdf(sample=N)` / `--sample N`: first, last, evenly spaced |
| document types | `text_based`, `scanned`, `image_based`, `mixed` (+ `empty`) with confidence |
| invisible OCR layers | detected via PDFium text render modes; *reused* in fast/edge profiles, re-OCR'd in others |
| text-quality detectors | `core/quality.py`, same detection classes and calibrated thresholds |
| OCR modes | `ocr="auto" | "force" | "off"` |
| PP-OCRv6 Small, pinned artifacts | the same pinned artifact set (oar-ocr v0.7.0, SHA-256) as TextLens's default engine |
| lazy runtime | no model load/download when no page needs OCR |
| per-page provenance | `PageProvenance` (source, model, revision, DPI, confidence, timings, warnings) |
| hosted fallback flag | generalised into TextLens's **fallback chain**: weak pages go to the next local or served model |
| position-aware extraction + reading order | `documents/pdf/native.py` + `documents/layout.py` (XY-cut with a tabular guard, akin to newspaper vs tabular) |
| rect/line table detection | ruling-line grid reconstruction (first strategy) |

## Where TextLens differs

- **Model-agnostic routing.** pdf-inspector hands weak pages to Firecrawl's
  hosted pipeline; TextLens routes each page among local and served models by
  page signals (tables, formulas, difficulty), hardware, budgets and license
  policy, with confidence-validated fallback.
- **Python + PDFium.** Inspection uses PDFium's object model (via pypdfium2,
  already a dependency) instead of a custom content-stream parser:
  text render modes, image bounds and pixel sizes, path segments, form
  XObjects and `FPDFText_HasUnicodeMapError`. Coverage is computed on a 48×48
  grid (union of image boxes), which handles tiled scans without a separate
  tiled-scan heuristic.
- **Post-extraction check.** After native extraction TextLens re-runs the
  quality detectors on the extracted text, catching garbled layers the object
  scan cannot see.
- **Beyond PDF.** The same page pipeline serves images, DOCX and PPTX, and a
  lazy `Document` API.
- **Fused pages.** `ocr_images=True` OCRs embedded images (screenshots) on
  native pages and merges them into reading order.

## Not (yet) adopted

- Heuristic (borderless) table detection and newspaper-layout OCR
  recommendation.
- Tagged-PDF structure tree roles (PDFium exposes them; not used yet).
- Right-to-left and CJK-specific extraction handling.
- Region-level native/OCR span fusion within a page.

## Tests

`tests/test_pdf_intelligence.py` generates PDFs with a dependency-free writer
(`tests/pdf_factory.py`): native text, JPEG scans of rendered text, invisible
OCR layers, form-wrapped scans, vector-outlined glyphs, a font whose
ToUnicode map permutes letters across cases (mimicking real broken CMaps),
ruled tables, two-column pages and rotated pages — and asserts kinds, reason
codes, reading order, tables and coordinates. `tests/test_routing_pipeline.py`
asserts that only the pages needing OCR reach an engine.
