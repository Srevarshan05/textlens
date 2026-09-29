# TextLens --- Current Architecture, Product Specification & TextLens 2.0 Roadmap

> **Document status:** Product / architecture planning document\
> **Project:** TextLens\
> **Purpose:** Define what TextLens already provides, what should become
> TextLens 2.0, and how to evolve it from a local OCR wrapper into a
> production-ready, model-agnostic OCR and Document AI infrastructure
> layer.\
> **Last updated:** 2026-09-29

------------------------------------------------------------------------

## 1. Executive Summary

TextLens started as a **local Python OCR/VLM framework** designed to
make modern OCR models easier to run locally.

The current system already provides the foundations of a useful OCR SDK:

-   Image OCR
-   PDF OCR
-   Multiple OCR/VLM model integrations
-   Model switching
-   Model caching
-   Batch OCR
-   GPU/CUDA detection
-   Hardware diagnostics
-   CPU fallback
-   Local inference
-   REST/FastAPI serving
-   CLI tooling
-   Multiple input forms such as local images, PDFs, URLs, Pillow
    objects and bytes
-   Structured/exportable results
-   JSON / Markdown / CSV / TXT style outputs
-   Model management concepts
-   Hardware-aware recommendations

The next version should **not** simply add more OCR models.

The strategic goal for TextLens 2.0 is:

> **One simple interface for OCR and document intelligence across edge
> devices, local machines, GPUs, servers and production deployments.**

The underlying OCR models should become interchangeable engines behind a
stable TextLens API.

The product should therefore evolve from:

``` text
TextLens = Python wrapper around OCR models
```

to:

``` text
TextLens = OCR infrastructure + model orchestration + document pipeline + deployment layer
```

The key differentiator should be the **engineering layer around OCR**:

-   automatic document inspection
-   intelligent model routing
-   hardware-aware execution
-   unified result schema
-   confidence and provenance
-   selective OCR
-   batching and streaming
-   caching
-   model lifecycle management
-   evaluation and benchmarking
-   production observability
-   REST/API/MCP interfaces
-   simple installation
-   excellent documentation
-   edge → local → server deployment continuity

------------------------------------------------------------------------

# 2. Product Vision

## 2.1 One-line positioning

> **TextLens is an open-source, model-agnostic OCR and Document AI
> runtime that automatically selects and orchestrates the right OCR
> pipeline for the task and available hardware.**

## 2.2 Developer promise

A developer should be able to install TextLens with:

``` bash
pip install textlens
```

and immediately run:

``` python
from textlens import OCR

ocr = OCR()

result = ocr("document.pdf")

print(result.text)
```

They should not need to understand:

-   CUDA compatibility
-   PyTorch versions
-   model-specific processors
-   model-specific prompt formats
-   OCR detector/recognizer plumbing
-   PDF rasterization details
-   GPU memory management
-   model download locations
-   batching strategies
-   backend differences

TextLens should handle these concerns.

Advanced users should still retain complete control.

------------------------------------------------------------------------

# 3. Current TextLens --- What Has Already Been Built

The current TextLens work has established a solid local OCR framework
foundation.

## 3.1 Core purpose

TextLens is designed around **local OCR and VLM inference**.

The framework abstracts away much of the repetitive setup required to
run OCR models locally.

The intended workflow is:

``` text
Input
  ↓
TextLens
  ↓
Hardware / model selection
  ↓
OCR / VLM inference
  ↓
Normalized result
  ↓
Application
```

------------------------------------------------------------------------

## 3.2 Supported input types

The current direction supports OCR from multiple sources:

-   local image paths
-   PDF files
-   URL-based inputs
-   raw bytes
-   Pillow images
-   document/image objects where supported by the backend

The long-term goal is to make input handling completely
backend-independent.

For example:

``` python
ocr("invoice.png")
```

``` python
ocr("paper.pdf")
```

``` python
ocr(image_bytes)
```

``` python
ocr(pil_image)
```

The developer should not need a different API for each source.

------------------------------------------------------------------------

# 4. Current OCR Capabilities

## 4.1 Image OCR

Basic image-to-text OCR is supported.

Example:

``` python
result = ocr("receipt.jpg")

print(result.text)
```

------------------------------------------------------------------------

## 4.2 PDF OCR

TextLens supports PDF OCR workflows.

The future architecture should expand this substantially by
distinguishing:

-   native-text PDFs
-   scanned PDFs
-   image-only PDFs
-   mixed PDFs
-   broken-encoding PDFs
-   partially scanned PDFs

This is one of the largest opportunities for TextLens 2.0.

------------------------------------------------------------------------

## 4.3 Multiple OCR/VLM models

TextLens was designed as a model wrapper/orchestration layer rather than
a single-model library.

Previous integrations and experiments have included models such as:

-   Qwen2.5-VL
-   GLM-OCR
-   LightOnOCR
-   HunyuanOCR
-   SmolVLM-class lightweight models
-   other OCR-capable Hugging Face VLMs

The exact model list should become a formal model registry in TextLens
2.0 instead of being scattered across implementation code.

------------------------------------------------------------------------

# 5. Current Model Management

TextLens already includes the concept of:

-   model switching
-   model caching
-   model selection
-   hardware-aware recommendations

This should evolve into a first-class model registry.

Future interface:

``` python
textlens.models.list()
```

``` python
textlens.models.info("glm-ocr")
```

``` bash
textlens models list
```

``` bash
textlens models install glm-ocr
```

------------------------------------------------------------------------

# 6. Current Hardware Detection

Hardware awareness is one of TextLens's strongest foundations.

The framework has already focused on:

-   CUDA detection
-   GPU availability
-   CPU fallback
-   hardware diagnostics
-   model suitability
-   local inference readiness

The project has also been tested across:

-   NVIDIA RTX GPUs
-   Jetson systems
-   CPU-only environments
-   CUDA/PyTorch configurations

This should become a formal hardware abstraction layer in TextLens 2.0.

------------------------------------------------------------------------

# 7. Current `doctor` Concept

TextLens already has a hardware/installation diagnostic workflow.

The next version should turn this into a polished command:

``` bash
textlens doctor
```

Example target output:

``` text
TextLens System Diagnostics
────────────────────────────────────

TextLens       2.0.x
Python         3.x
OS             Ubuntu 24.04

CPU            Detected
RAM            32 GB

GPU            NVIDIA RTX 3060
VRAM           12 GB
CUDA           Available
PyTorch        Available

Backends
  CPU          ✓
  CUDA         ✓
  ONNX         ✓
  Paddle       optional
  TensorRT     optional
  vLLM         optional

Recommended profile:
  balanced

Recommended model:
  GLM-OCR
```

This should become one of the main ways TextLens solves the "ML
installation nightmare."

------------------------------------------------------------------------

# 8. Current Batch OCR

TextLens already includes batch OCR capabilities.

The next version should turn batching into a real processing engine:

``` text
Input queue
    ↓
Preprocessing workers
    ↓
Dynamic batch collector
    ↓
GPU inference
    ↓
Post-processing
    ↓
Result writer
```

Required capabilities:

-   dynamic batching
-   worker pools
-   backpressure
-   retries
-   checkpointing
-   progress reporting
-   failed-document recovery
-   memory-aware batching
-   per-page batching for PDFs

------------------------------------------------------------------------

# 9. Current Caching

TextLens already has model/result caching concepts.

TextLens 2.0 should distinguish:

### Model cache

``` text
~/.cache/textlens/models/
```

### Result cache

``` text
~/.cache/textlens/results/
```

### Preprocessing cache

``` text
~/.cache/textlens/preprocessed/
```

Documents should be identified using content hashes where appropriate.

Repeated processing should therefore become:

``` text
Document
   ↓
Hash
   ↓
Cache lookup
   ↓
Existing result?
   ├── yes → return
   └── no → process
```

------------------------------------------------------------------------

# 10. Current Serving/API Direction

TextLens already includes local serving/API capabilities.

The next version should formalize this into a production server:

``` bash
textlens serve
```

Potential API:

``` text
POST /ocr
POST /document
POST /batch
POST /extract

GET /models
GET /health
GET /metrics
GET /jobs/{id}
```

------------------------------------------------------------------------

# 11. Current CLI Direction

The CLI should become a major part of the project rather than merely a
debugging utility.

Target:

``` bash
textlens ocr image.png
```

``` bash
textlens document paper.pdf
```

``` bash
textlens batch ./documents/
```

``` bash
textlens inspect document.pdf
```

``` bash
textlens models list
```

``` bash
textlens models install glm-ocr
```

``` bash
textlens doctor
```

``` bash
textlens benchmark ./dataset
```

``` bash
textlens serve
```

------------------------------------------------------------------------

# 12. Current Package / Distribution History

TextLens has gone through early packaging iterations including:

-   TestPyPI experimentation
-   `textlens-srevarshan`
-   `textlens-ocr`
-   `textlens`

The next major release should consolidate the package identity around:

``` text
textlens
```

and make installation predictable.

------------------------------------------------------------------------

# 13. The Biggest Problem With TextLens Today

The current architecture is useful, but it is still too close to:

> "A wrapper around multiple OCR models."

That is not enough to create a long-term developer ecosystem.

There are already many excellent OCR projects and model repositories.

TextLens should therefore **not compete by having the largest number of
models**.

Instead:

> **TextLens should normalize and orchestrate the OCR ecosystem.**

The models become engines.

TextLens becomes the infrastructure.

------------------------------------------------------------------------

# 14. TextLens 2.0 --- Strategic Direction

TextLens 2.0 should be organized around seven major pillars:

``` text
1. Simple Installation
2. Intelligent OCR Routing
3. Unified OCR Result Model
4. Hardware-Aware Runtime
5. Production Processing Engine
6. Developer/Agent Interfaces
7. Evaluation + Observability
```

------------------------------------------------------------------------

# 15. Pillar 1 --- Installation Must Be Easy

## 15.1 Default install

The default experience should be:

``` bash
pip install textlens
```

Then:

``` python
from textlens import OCR

ocr = OCR()
result = ocr("image.png")
```

No CUDA setup should be required for the basic path.

------------------------------------------------------------------------

## 15.2 Optional capabilities

Use optional dependencies:

``` bash
pip install "textlens[gpu]"
pip install "textlens[edge]"
pip install "textlens[documents]"
pip install "textlens[vlm]"
pip install "textlens[server]"
pip install "textlens[dev]"
```

Potential combined installation:

``` bash
pip install "textlens[all]"
```

But `all` must remain carefully curated.

Do not force users to install every heavyweight ML runtime.

------------------------------------------------------------------------

# 16. Model Downloads Must Be Separate From the Python Package

The Python package should remain small.

Models should be downloaded separately:

``` bash
textlens models install ppocrv6-small
```

or automatically:

``` python
ocr = OCR(model="glm-ocr")
```

If the model is unavailable:

``` text
Model 'glm-ocr' is not installed.

Download model? [Y/n]
```

This is much better than shipping multi-gigabyte model weights through
PyPI.

------------------------------------------------------------------------

# 17. `textlens setup`

Provide an interactive setup command:

``` bash
textlens setup
```

Possible flow:

``` text
Welcome to TextLens

How do you plan to use TextLens?

1. CPU / Edge
2. NVIDIA GPU
3. Document AI
4. Server
5. Development / Everything

Select:
```

Then automatically inspect:

-   CPU
-   RAM
-   GPU
-   VRAM
-   CUDA
-   operating system
-   Python
-   available backends

and recommend a profile.

------------------------------------------------------------------------

# 18. Pillar 2 --- Intelligent OCR Routing

This should become the central feature.

Instead of:

``` python
OCR(model="some-model")
```

the default should be:

``` python
OCR(mode="auto")
```

The routing pipeline:

``` text
Input
 ↓
Document Analyzer
 ↓
Task Detection
 ↓
Hardware Detection
 ↓
Installed Model Registry
 ↓
Model Scoring
 ↓
Selected Pipeline
 ↓
OCR
 ↓
Validation
 ↓
Unified Result
```

------------------------------------------------------------------------

# 19. PDF Intelligence

TextLens should inspect PDFs before performing OCR.

Inspired by the excellent engineering approach in Firecrawl's
`pdf-inspector`, TextLens should detect:

-   text-based PDFs
-   scanned PDFs
-   image-based PDFs
-   mixed PDFs
-   pages requiring OCR
-   broken font encoding
-   pages containing likely tables
-   pages with multiple columns

Firecrawl's current project classifies PDFs in roughly 10--50 ms,
identifies pages requiring OCR, extracts native text with position
awareness, and supports selective PP-OCRv6 OCR rather than blindly
OCRing every page. This is exactly the type of routing architecture
TextLens should learn from.

Reference:

https://github.com/firecrawl/pdf-inspector

------------------------------------------------------------------------

# 20. Selective OCR

Never OCR an entire PDF blindly.

Target architecture:

``` text
PDF
 │
 ├── Page 1 → native text → no OCR
 ├── Page 2 → native text → no OCR
 ├── Page 3 → scan → OCR
 ├── Page 4 → scan → OCR
 ├── Page 5 → native text → no OCR
 └── Page 6 → broken encoding → OCR
```

This provides:

-   lower latency
-   lower GPU usage
-   lower energy consumption
-   lower cost
-   better production throughput

------------------------------------------------------------------------

# 21. Smart Model Profiles

Users should not need to memorize model names.

Profiles:

``` text
edge
fast
balanced
accurate
document
table
multilingual
handwriting
server
auto
```

Example:

``` python
OCR(profile="edge")
```

could choose a small OCR model.

``` python
OCR(profile="balanced")
```

could choose a medium local OCR/VLM.

``` python
OCR(profile="accurate")
```

could select a heavier document model.

------------------------------------------------------------------------

# 22. But Power Users Must Have Direct Model Control

Advanced usage:

``` python
OCR(
    model="deepseek-ocr-2"
)
```

or:

``` python
OCR(
    backend="transformers",
    model="some-huggingface-model"
)
```

TextLens should therefore support both:

``` text
High-level profiles
        +
Explicit model selection
```

------------------------------------------------------------------------

# 23. Model Registry

Create a formal registry.

Example metadata:

``` yaml
name: glm-ocr
provider: zai
parameters: 0.9B

capabilities:
  text: true
  layout: true
  table: true
  formula: true
  handwriting: false

hardware:
  cpu: false
  cuda: true

memory:
  minimum_vram: 4GB

outputs:
  text: true
  markdown: true
  json: true
```

The router can then select models programmatically.

------------------------------------------------------------------------

# 24. Pillar 3 --- Unified Result Schema

Every model should produce the same result abstraction.

Example:

``` python
result.text
result.pages
result.blocks
result.words
result.tables
result.layout
result.confidence
result.metadata
result.provenance
```

A model-specific output should never leak unnecessarily into the public
API.

------------------------------------------------------------------------

# 25. Bounding Boxes

Every recognized object should be able to expose:

``` json
{
  "text": "Total Amount: ₹42,500",
  "confidence": 0.97,
  "bbox": [420, 830, 710, 870],
  "page": 3,
  "model": "glm-ocr"
}
```

This enables downstream:

-   document search
-   highlighting
-   annotation
-   RAG citations
-   visual debugging
-   structured extraction

------------------------------------------------------------------------

# 26. Provenance

Every output should be traceable.

Example:

``` text
Document
  ↓
Page 7
  ↓
Region 14
  ↓
Model: GLM-OCR
  ↓
Confidence: 0.96
  ↓
Bounding box
  ↓
Original image
```

This is essential for serious document workflows.

------------------------------------------------------------------------

# 27. Confidence-Aware Processing

TextLens should not treat all OCR text equally.

Example:

``` text
Confidence >= 0.95
    → accept

0.80–0.95
    → optional validation

< 0.80
    → fallback model / retry
```

Future:

``` python
OCR(profile="reliable")
```

could automatically perform fallback processing.

------------------------------------------------------------------------

# 28. Multi-Model Fallback

Example:

``` text
GLM-OCR
   ↓
confidence < threshold
   ↓
DeepSeek-OCR-2
   ↓
confidence improved?
   ↓
return best result
```

This makes the framework much more robust for production.

------------------------------------------------------------------------

# 29. Pillar 4 --- Hardware-Aware Runtime

TextLens should support the same API across:

### Edge

-   Raspberry Pi
-   Jetson
-   CPU systems
-   low-memory devices

### Local

-   NVIDIA RTX GPUs
-   Apple Silicon where supported
-   workstation GPUs

### Server

-   single GPU
-   multi-GPU
-   vLLM
-   containers

### Remote

-   HTTP model endpoints
-   OpenAI-compatible endpoints
-   internal inference services

------------------------------------------------------------------------

# 30. Backend Abstraction

Architecture:

``` text
                 TextLens Core
                      │
        ┌─────────────┼─────────────┐
        │             │             │
       ONNX        PyTorch        Paddle
        │             │             │
     TensorRT       vLLM        Remote API
```

A model adapter should hide backend-specific details.

------------------------------------------------------------------------

# 31. Edge Backend

The edge stack should prioritize:

-   tiny models
-   ONNX
-   TensorRT
-   quantization
-   low memory
-   low latency
-   CPU fallback

Potential model families:

``` text
PP-OCRv6 Tiny
PP-OCRv6 Small
other compact OCR models
```

------------------------------------------------------------------------

# 32. Local GPU Backend

For workstation users:

``` text
RTX 3060
RTX 4060
RTX 4070
RTX 4090
etc.
```

TextLens should automatically estimate:

-   available VRAM
-   model requirements
-   batch size
-   quantization

------------------------------------------------------------------------

# 33. Server Backend

For production:

``` text
TextLens API
     ↓
Queue
     ↓
Workers
     ↓
Dynamic batching
     ↓
Inference server
```

vLLM should be an optional high-throughput backend, not a mandatory
dependency.

------------------------------------------------------------------------

# 34. Production Architecture Inspiration

The Neural Maze Production OCR Course is particularly useful as a
systems-design reference.

Its current open-source course demonstrates:

-   Kubernetes
-   AKS/GKE
-   GPU node pools
-   Qwen 3.5 4B
-   GLM-OCR / PP-DocLayoutV3
-   vLLM
-   continuous batching
-   PagedAttention
-   Multi-Token Prediction
-   Redis
-   Rust/Axum ingestion
-   asynchronous workers
-   `/dev/shm` zero-copy handoff
-   KEDA autoscaling
-   API gateways
-   JWT/rate limiting
-   MCP integration

The course reports a 1.86 pages/second generative OCR throughput in its
particular deployment and demonstrates independent scaling of layout and
inference workloads. These numbers should be treated as
deployment-specific rather than universal TextLens benchmarks.

Source:

https://github.com/neural-maze/production-ocr-course

------------------------------------------------------------------------

# 35. What TextLens Should Learn From That Course

TextLens should not copy the entire Kubernetes stack into the default
package.

Instead:

``` text
TextLens Core
       ↓
simple local processing
       ↓
production adapters
       ↓
optional queue / worker architecture
       ↓
optional Kubernetes deployment
```

The same OCR API should work at every level.

------------------------------------------------------------------------

# 36. Small Project → Production Continuity

This should become one of TextLens's strongest principles.

### Student project

``` python
ocr = OCR()
result = ocr("image.png")
```

### Small application

``` python
ocr = OCR(profile="balanced")
```

### Backend API

``` bash
textlens serve
```

### Production

``` text
API Gateway
    ↓
Queue
    ↓
TextLens workers
    ↓
vLLM / inference backend
    ↓
GPU cluster
```

The application does not need to rewrite its OCR code.

------------------------------------------------------------------------

# 37. Pillar 5 --- Document Intelligence

TextLens should grow beyond:

``` text
image → text
```

into:

``` text
document → structured understanding
```

Public APIs should eventually include:

``` python
ocr.text(...)
ocr.layout(...)
ocr.tables(...)
ocr.formulas(...)
ocr.handwriting(...)
ocr.extract(...)
ocr.classify(...)
ocr.chunk(...)
```

------------------------------------------------------------------------

# 38. Structured Extraction

Example:

``` python
result = ocr.extract(
    "invoice.pdf",
    schema={
        "vendor": "string",
        "invoice_number": "string",
        "date": "date",
        "total": "float"
    }
)
```

Output:

``` json
{
  "vendor": "...",
  "invoice_number": "...",
  "date": "...",
  "total": 42500.0
}
```

This makes TextLens useful for real applications.

------------------------------------------------------------------------

# 39. Table Extraction

Table support should be first-class.

``` python
result.tables
```

Each table should support:

-   cells
-   rows
-   columns
-   bounding boxes
-   confidence
-   page
-   Markdown
-   CSV
-   JSON

------------------------------------------------------------------------

# 40. Formula Extraction

For scientific documents:

``` python
result.formulas
```

Potential outputs:

-   LaTeX
-   bounding box
-   page number
-   confidence

------------------------------------------------------------------------

# 41. Layout

The layout layer should recognize:

-   paragraphs
-   headings
-   tables
-   figures
-   formulas
-   captions
-   lists
-   headers
-   footers
-   signatures
-   forms

------------------------------------------------------------------------

# 42. RAG Integration

TextLens should be directly useful for RAG.

Example:

``` python
chunks = result.to_chunks(
    strategy="semantic",
    max_tokens=500
)
```

Output should preserve:

-   page
-   bounding box
-   source
-   section
-   document ID

This enables citation-aware RAG.

------------------------------------------------------------------------

# 43. RAG-Friendly Markdown

Provide:

``` python
result.to_markdown()
```

with structure such as:

``` markdown
# Section

Text...

## Table

| Item | Amount |
|---|---:|
| A | 100 |

<!-- page: 4 -->
```

Page/source markers should be optionally preserved.

------------------------------------------------------------------------

# 44. Agent / MCP Support

MCP should be a first-class integration in the production roadmap.

Potential tools:

``` text
ocr_image
ocr_document
extract_table
extract_text
inspect_document
search_document
extract_schema
get_page
```

This allows coding agents and AI assistants to use TextLens as an OCR
tool.

The Neural Maze production course also demonstrates an MCP wrapper
around its OCR pipeline, reinforcing the usefulness of making OCR
agent-accessible.

------------------------------------------------------------------------

# 45. REST API

Target:

``` text
POST /ocr
POST /document
POST /extract
POST /batch

GET /models
GET /health
GET /metrics
GET /jobs/{id}
```

------------------------------------------------------------------------

# 46. OpenAI-Compatible Interface

A useful optional server mode:

``` bash
textlens serve --api openai
```

This allows applications already designed around OpenAI-compatible
interfaces to use TextLens as a local/private OCR service.

------------------------------------------------------------------------

# 47. Streaming

Large PDFs should not require loading the complete result into memory.

Target:

``` python
for page in ocr.stream("large.pdf"):
    print(page.text)
```

This should support:

-   page streaming
-   result callbacks
-   async processing
-   cancellation
-   progress events

------------------------------------------------------------------------

# 48. Batch Engine

Production batch processing:

``` text
Input
 ↓
Queue
 ↓
CPU preprocessing
 ↓
Dynamic batcher
 ↓
GPU
 ↓
Post-processing
 ↓
Result writer
```

Features:

-   dynamic batch sizing
-   retry
-   checkpointing
-   backpressure
-   worker pools
-   progress reporting
-   partial results

------------------------------------------------------------------------

# 49. Zero-Copy / Shared Memory Optimization

For high-throughput deployments, consider shared-memory handoff.

Concept:

``` text
Rasterization
      ↓
/dev/shm
      ↓
Layout engine
      ↓
Shared image buffers
      ↓
Inference engine
```

This should be optional and platform-specific.

It is especially relevant for large PDFs and high-concurrency server
deployments.

------------------------------------------------------------------------

# 50. Production Queue Architecture

For very large deployments:

``` text
Client
  ↓
API
  ↓
Queue
  ↓
OCR workers
  ↓
Inference server
  ↓
Result store
```

Potential integrations:

-   Redis
-   Kafka
-   RabbitMQ
-   cloud queues

Do not make any queue mandatory for local users.

------------------------------------------------------------------------

# 51. Autoscaling

Future deployment templates could support:

``` text
Kubernetes
KEDA
GPU node pools
scale-to-zero
queue-depth scaling
```

This should live in a separate deployment package/repository or optional
deployment directory.

It should never inflate the core Python package.

------------------------------------------------------------------------

# 52. Security

Production mode should support:

-   API keys
-   JWT
-   rate limiting
-   request size limits
-   allowed MIME types
-   file type validation
-   temporary-file cleanup
-   configurable retention
-   no-persistence mode
-   audit logging
-   encrypted transport
-   private network deployment

Security should be documented as a dedicated section, not hidden in
implementation details.

------------------------------------------------------------------------

# 53. Privacy

TextLens should emphasize local/private OCR.

Core promise:

``` text
Your document does not have to leave your machine.
```

This is especially important for:

-   legal documents
-   internal company documents
-   research
-   financial records
-   confidential forms

Cloud/remote backends should be explicitly opt-in.

------------------------------------------------------------------------

# 54. Evaluation Framework

This should be one of the most innovative parts of TextLens.

Developers should be able to compare OCR engines on their own data:

``` bash
textlens benchmark ./dataset
```

Example:

``` text
Model             CER      WER      Latency     VRAM
-----------------------------------------------------
PP-OCRv6          4.2%     7.1%     82 ms       1.1 GB
Surya-2           2.8%     4.9%     310 ms      3.2 GB
GLM-OCR           2.1%     3.7%     540 ms      4.8 GB
DeepSeek-OCR-2    1.8%     3.2%     1.2 s       8.1 GB
```

The values above are illustrative, not benchmark claims.

------------------------------------------------------------------------

# 55. Benchmark Metrics

Support:

### Text

-   CER
-   WER
-   exact match
-   normalized edit distance

### Layout

-   IoU
-   reading-order metrics

### Tables

-   TEDS
-   cell accuracy
-   structure accuracy

### System

-   latency
-   throughput
-   VRAM
-   RAM
-   CPU utilization
-   GPU utilization
-   energy where measurable

------------------------------------------------------------------------

# 56. Dataset Evaluation

Users should be able to provide:

``` text
dataset/
    images/
    ground_truth/
```

and run:

``` bash
textlens benchmark \
    --dataset ./dataset \
    --models auto
```

The result should recommend a model based on measured performance rather
than marketing claims.

------------------------------------------------------------------------

# 57. Profiling

Add:

``` bash
textlens profile document.pdf
```

Output:

``` text
Preprocessing       82 ms
Model loading       0 ms
Inference           412 ms
Postprocessing      39 ms
Total               533 ms

Peak VRAM           4.6 GB
Peak RAM            2.1 GB
```

------------------------------------------------------------------------

# 58. Visual Debugging

A major developer feature:

``` bash
textlens inspect document.pdf
```

Generate visual overlays for:

-   text boxes
-   tables
-   layout regions
-   reading order
-   confidence
-   model assignment

This should be one of the best debugging tools in the project.

------------------------------------------------------------------------

# 59. Model Routing Explainability

When `mode="auto"` is used, TextLens should explain its decision.

Example:

``` text
Selected: GLM-OCR

Reason:
  Document type: scanned PDF
  Tables detected: 3
  Formula regions: 2
  GPU available: RTX 3060 12GB
  Estimated VRAM: 4.8GB
  Installed model compatibility: ✓

Fallback:
  DeepSeek-OCR-2
```

This makes automatic routing trustworthy.

------------------------------------------------------------------------

# 60. New Innovative Feature --- OCR Cost/Latency Budget

Let developers specify a budget:

``` python
OCR(
    latency_budget_ms=500
)
```

or:

``` python
OCR(
    memory_budget_gb=4
)
```

or:

``` python
OCR(
    quality="high",
    max_latency_ms=1000
)
```

TextLens then selects the best available pipeline.

This turns model selection into an optimization problem:

``` text
Quality
Latency
Memory
Cost
Hardware
Task
```

------------------------------------------------------------------------

# 61. New Innovative Feature --- Adaptive OCR

Instead of deciding the model only once:

``` text
Page
 ↓
Fast OCR
 ↓
Confidence
 ├── high → done
 └── low
      ↓
      better model
      ↓
      done
```

This can dramatically reduce expensive inference.

Simple pages stay cheap.

Difficult pages receive expensive processing.

------------------------------------------------------------------------

# 62. New Innovative Feature --- Document Difficulty Score

TextLens could estimate:

``` text
difficulty = 0.82
```

based on:

-   image quality
-   text density
-   rotation
-   number of columns
-   handwriting
-   tables
-   formulas
-   language
-   layout complexity
-   OCR confidence

Then routing becomes:

``` text
difficulty < 0.3 → lightweight OCR
0.3–0.7          → balanced
> 0.7            → advanced VLM
```

------------------------------------------------------------------------

# 63. New Innovative Feature --- Automatic Pipeline Generation

Given:

``` bash
textlens analyze dataset/
```

TextLens could report:

``` text
Dataset profile

82% printed documents
9% tables
5% handwriting
4% formulas

Recommended pipeline:

PP-OCRv6 → 82%
GLM-OCR → tables
advanced VLM → handwriting/formulas
```

This is much more useful than a static model list.

------------------------------------------------------------------------

# 64. New Innovative Feature --- Model Cards Inside TextLens

Every model should expose:

``` bash
textlens models info glm-ocr
```

including:

-   parameter count
-   license
-   languages
-   supported tasks
-   VRAM estimate
-   CPU support
-   GPU support
-   quantization options
-   backend
-   benchmark results
-   known limitations
-   source repository

This makes model discovery much easier.

------------------------------------------------------------------------

# 65. New Innovative Feature --- Reproducible OCR

Production results should contain a pipeline fingerprint:

``` json
{
  "textlens_version": "2.0.0",
  "model": "glm-ocr",
  "model_revision": "...",
  "backend": "transformers",
  "device": "cuda:0",
  "config_hash": "...",
  "document_hash": "..."
}
```

This lets users reproduce results later.

------------------------------------------------------------------------

# 66. New Innovative Feature --- OCR Artifacts

Instead of returning only text, allow:

``` text
result/
    document.json
    document.md
    pages/
    images/
    tables/
    layout.json
    provenance.json
    metadata.json
```

This is excellent for pipelines.

------------------------------------------------------------------------

# 67. New Innovative Feature --- Document Search Index

Potential API:

``` python
index = textlens.index("./documents")

results = index.search("invoice total")
```

TextLens could use its own OCR results and optionally connect to vector
databases.

This turns OCR into a document ingestion layer.

------------------------------------------------------------------------

# 68. New Innovative Feature --- Human Review Mode

For sensitive workflows:

``` text
OCR
 ↓
Confidence
 ↓
Low-confidence regions
 ↓
Human review
 ↓
Correction
 ↓
Ground truth
```

Corrections could optionally be exported as datasets for future model
evaluation.

This creates a feedback loop:

``` text
OCR → Review → Dataset → Benchmark → Better Routing
```

------------------------------------------------------------------------

# 69. New Innovative Feature --- Synthetic OCR Evaluation

TextLens could eventually generate controlled evaluation datasets
involving:

-   blur
-   rotation
-   compression
-   low resolution
-   skew
-   noise
-   lighting changes

Then compare models.

This would make TextLens useful to OCR engineers, not just application
developers.

------------------------------------------------------------------------

# 70. New Innovative Feature --- OCR Playground

A web UI:

``` text
Upload document
       ↓
Choose model/profile
       ↓
Run OCR
       ↓
Compare outputs
       ↓
Visual bounding boxes
       ↓
Latency / VRAM
       ↓
Export
```

This would be an excellent GitHub demo.

------------------------------------------------------------------------

# 71. Documentation Strategy

The documentation should be treated as a product.

Recommended structure:

``` text
docs/
├── getting-started/
│   ├── installation.md
│   ├── first-ocr.md
│   └── troubleshooting.md
│
├── concepts/
│   ├── architecture.md
│   ├── routing.md
│   ├── models.md
│   ├── results.md
│   └── backends.md
│
├── tasks/
│   ├── text.md
│   ├── pdf.md
│   ├── tables.md
│   ├── formulas.md
│   ├── layout.md
│   └── extraction.md
│
├── deployment/
│   ├── cpu.md
│   ├── gpu.md
│   ├── jetson.md
│   ├── docker.md
│   ├── server.md
│   └── kubernetes.md
│
├── production/
│   ├── scaling.md
│   ├── batching.md
│   ├── caching.md
│   ├── observability.md
│   └── security.md
│
├── models/
│   ├── registry.md
│   ├── model-selection.md
│   └── compatibility.md
│
└── evaluation/
    ├── benchmarks.md
    ├── datasets.md
    └── profiling.md
```

------------------------------------------------------------------------

# 72. Documentation Levels

Documentation should serve three audiences.

## Beginner

``` text
pip install textlens
→ OCR image
→ OCR PDF
→ export Markdown
```

## Developer

``` text
model selection
custom pipeline
batch processing
REST API
```

## Production Engineer

``` text
vLLM
Kubernetes
queue architecture
autoscaling
observability
security
```

One project should serve all three.

------------------------------------------------------------------------

# 73. Production OCR Course as Documentation Inspiration

The Neural Maze course is strong because it explains **why** system
components exist rather than simply providing commands.

Its structure covers:

-   architecture
-   model selection
-   throughput
-   GPU scheduling
-   batching
-   queueing
-   scaling
-   networking
-   security
-   deployment

TextLens documentation should adopt the same principle:

> **Don't only document what a function does. Explain when and why a
> developer should use it.**

Source:

https://github.com/neural-maze/production-ocr-course

------------------------------------------------------------------------

# 74. TextLens Documentation Should Include Engineering Guides

Examples:

### Why selective OCR?

### How PDF routing works

### Why dynamic batching matters

### Why model quantization matters

### How VRAM affects model selection

### How to run OCR on Jetson

### How to build a private OCR API

### How to process 100,000 PDFs

### How to benchmark OCR models

### How to debug low OCR confidence

### How to build an OCR + RAG system

This makes the project educational as well as useful.

------------------------------------------------------------------------

# 75. Examples Repository

Create:

``` text
examples/
├── basic/
├── pdf/
├── tables/
├── rag/
├── fastapi/
├── batch/
├── jetson/
├── docker/
├── vllm/
├── kubernetes/
├── mcp/
└── benchmarking/
```

Every example should be runnable.

------------------------------------------------------------------------

# 76. Production Reference Architectures

Include diagrams for:

### Local

``` text
Application
   ↓
TextLens
   ↓
Local GPU
```

### API

``` text
Client
 ↓
FastAPI
 ↓
TextLens
 ↓
GPU
```

### High-throughput

``` text
Client
 ↓
Gateway
 ↓
Queue
 ↓
Workers
 ↓
Inference server
 ↓
Result store
```

### Kubernetes

``` text
API
 ↓
Queue
 ↓
CPU preprocessing
 ↓
GPU inference
 ↓
Result store
```

------------------------------------------------------------------------

# 77. Testing Strategy

TextLens 2.0 should have:

### Unit tests

-   routing
-   result schema
-   caching
-   model registry
-   input handling

### Integration tests

-   OCR backends
-   PDF pipelines
-   API
-   CLI

### Golden tests

Store expected OCR outputs for representative documents.

### Performance tests

Track:

-   latency
-   throughput
-   memory
-   VRAM

------------------------------------------------------------------------

# 78. CI/CD

Every pull request should ideally run:

``` text
lint
type check
unit tests
integration tests
API tests
CLI tests
documentation build
```

GPU tests can run separately.

------------------------------------------------------------------------

# 79. Compatibility Matrix

Maintain a visible matrix:

  Platform     CPU   NVIDIA   Jetson   OCR                 VLM
  ---------- ----- -------- -------- ----- -------------------
  Linux          ✓        ✓        ✓     ✓                   ✓
  Windows        ✓        ✓      ---     ✓                   ✓
  macOS          ✓      ---      ---     ✓   backend-dependent

The exact support should be verified continuously rather than promised
permanently.

------------------------------------------------------------------------

# 80. Dependency Philosophy

The most important dependency rule:

> **The core should stay boring.**

Avoid forcing:

-   TensorRT
-   Paddle
-   vLLM
-   CUDA libraries
-   massive model weights
-   Kubernetes clients

onto every user.

Optional functionality belongs behind extras/plugins/backends.

------------------------------------------------------------------------

# 81. Versioning Strategy

Suggested:

``` text
0.x = experimental architecture
1.0 = stable developer API
1.x = compatibility releases
2.x = major runtime/document intelligence evolution
```

For TextLens 2.0, stabilize:

-   `OCR`
-   `Document`
-   `Result`
-   `ModelRegistry`
-   `Backend`
-   `Router`

before adding dozens of new APIs.

------------------------------------------------------------------------

# 82. Proposed Core API

``` python
from textlens import OCR

ocr = OCR()

result = ocr("document.pdf")
```

Advanced:

``` python
ocr = OCR(
    profile="balanced",
    device="auto"
)
```

Explicit:

``` python
ocr = OCR(
    model="glm-ocr",
    backend="transformers"
)
```

Streaming:

``` python
for page in ocr.stream("large.pdf"):
    ...
```

Batch:

``` python
results = ocr.batch("./documents/")
```

Extraction:

``` python
result = ocr.extract(
    "invoice.pdf",
    schema=invoice_schema
)
```

------------------------------------------------------------------------

# 83. Proposed Result API

``` python
result.text
result.pages
result.blocks
result.words
result.tables
result.formulas
result.layout
result.confidence
result.provenance
result.metadata
```

Conversions:

``` python
result.to_json()
result.to_markdown()
result.to_html()
result.to_csv()
result.to_text()
```

------------------------------------------------------------------------

# 84. Proposed Document API

``` python
document = ocr.load("paper.pdf")
```

Then:

``` python
document.pages
document.text()
document.tables()
document.layout()
document.images()
document.metadata()
document.to_markdown()
document.to_json()
document.chunks()
```

------------------------------------------------------------------------

# 85. Proposed Model API

``` python
textlens.models.list()
textlens.models.search("table")
textlens.models.info("glm-ocr")
textlens.models.install("glm-ocr")
textlens.models.remove("glm-ocr")
```

------------------------------------------------------------------------

# 86. Proposed CLI

``` bash
textlens ocr image.png
textlens document paper.pdf
textlens batch ./documents
textlens inspect paper.pdf
textlens extract invoice.pdf
textlens benchmark ./dataset
textlens models list
textlens models install glm-ocr
textlens doctor
textlens setup
textlens serve
```

------------------------------------------------------------------------

# 87. Proposed Architecture

``` text
textlens/
│
├── core/
│   ├── engine.py
│   ├── router.py
│   ├── registry.py
│   ├── pipeline.py
│   ├── result.py
│   └── config.py
│
├── input/
│   ├── image.py
│   ├── pdf.py
│   └── document.py
│
├── analysis/
│   ├── pdf_classifier.py
│   ├── document_classifier.py
│   ├── difficulty.py
│   └── layout.py
│
├── backends/
│   ├── pytorch.py
│   ├── transformers.py
│   ├── onnx.py
│   ├── paddle.py
│   ├── tensorrt.py
│   ├── vllm.py
│   └── remote.py
│
├── models/
│   ├── registry.py
│   ├── ppocr.py
│   ├── surya.py
│   ├── glm.py
│   ├── deepseek.py
│   └── ...
│
├── tasks/
│   ├── text.py
│   ├── tables.py
│   ├── formulas.py
│   ├── layout.py
│   ├── handwriting.py
│   └── extraction.py
│
├── runtime/
│   ├── batching.py
│   ├── caching.py
│   ├── memory.py
│   ├── workers.py
│   └── streaming.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── metrics.py
│   └── profiler.py
│
├── serving/
│   ├── fastapi.py
│   ├── mcp.py
│   └── openai.py
│
└── cli/
    └── main.py
```

------------------------------------------------------------------------

# 88. TextLens 2.0 Development Phases

## Phase 1 --- Core Stabilization

Priority:

-   unified `OCR`
-   unified `Result`
-   model registry
-   backend abstraction
-   installation cleanup
-   `doctor`
-   configuration system
-   logging
-   error handling

Goal:

> Make the foundation reliable.

------------------------------------------------------------------------

## Phase 2 --- Intelligent Routing

Implement:

-   PDF classification
-   page-level routing
-   hardware-aware model selection
-   model profiles
-   confidence-aware fallback
-   document difficulty estimation

Goal:

> Make TextLens smart.

------------------------------------------------------------------------

## Phase 3 --- Document Intelligence

Implement:

-   layout
-   tables
-   formulas
-   structured extraction
-   Markdown
-   JSON
-   provenance
-   RAG chunks

Goal:

> Make TextLens useful beyond plain OCR.

------------------------------------------------------------------------

## Phase 4 --- Performance

Implement:

-   batching
-   streaming
-   cache
-   memory management
-   quantization support
-   ONNX
-   TensorRT where practical
-   vLLM backend

Goal:

> Make TextLens fast.

------------------------------------------------------------------------

## Phase 5 --- Serving

Implement:

-   FastAPI
-   asynchronous jobs
-   health endpoints
-   metrics
-   API authentication
-   rate limits
-   Docker

Goal:

> Make TextLens deployable.

------------------------------------------------------------------------

## Phase 6 --- Production Scale

Implement deployment reference architectures:

-   Redis queue
-   workers
-   GPU inference
-   Kubernetes
-   KEDA
-   autoscaling
-   shared-memory handoff
-   multi-GPU routing

Goal:

> Make TextLens production-capable without making the core package
> complicated.

------------------------------------------------------------------------

## Phase 7 --- Evaluation Ecosystem

Implement:

-   benchmark runner
-   model comparison
-   profiling
-   datasets
-   golden tests
-   visual evaluation
-   regression detection

Goal:

> Make TextLens measurable.

------------------------------------------------------------------------

# 89. What NOT to Build in the Core

Do not put these directly into the core package:

-   Kubernetes client
-   Redis client
-   cloud SDKs
-   every ML framework
-   every OCR model
-   giant model weights
-   enterprise gateway
-   vector database
-   complex web dashboard

Instead, provide:

``` text
Core
+
Optional backends
+
Optional deployment packages
+
Reference architectures
```

This keeps installation fast and stable.

------------------------------------------------------------------------

# 90. How TextLens Can Gain GitHub Stars Organically

The project should be useful before someone even reads the source.

The GitHub README should immediately show:

``` text
pip install textlens

from textlens import OCR

ocr = OCR()
result = ocr("invoice.pdf")

print(result.text)
```

Then show:

``` text
✓ Edge
✓ CPU
✓ NVIDIA GPU
✓ PDF
✓ Images
✓ Tables
✓ Layout
✓ VLM OCR
✓ Batch
✓ REST API
✓ MCP
✓ RAG
✓ Benchmarking
```

Then a visual architecture diagram.

Then benchmark results.

Then model compatibility.

Then production deployment.

------------------------------------------------------------------------

# 91. Star-Worthy Features

The features most likely to make developers genuinely remember TextLens
are not simply model count.

They are:

1.  **Zero-friction installation**
2.  **Automatic model routing**
3.  **PDF selective OCR**
4.  **Hardware-aware inference**
5.  **Unified result schema**
6.  **Confidence-aware fallback**
7.  **Visual OCR inspection**
8.  **Built-in benchmarking**
9.  **Model manager**
10. **One API from edge to server**
11. **RAG-ready output**
12. **MCP support**
13. **OpenAI-compatible API**
14. **Production reference architecture**
15. **Excellent documentation**

------------------------------------------------------------------------

# 92. TextLens's Long-Term Position

The ecosystem already contains:

-   individual OCR models
-   OCR model libraries
-   document parsers
-   PDF extraction libraries
-   inference servers
-   cloud OCR APIs

TextLens should connect these pieces.

The positioning should be:

``` text
                    OCR ECOSYSTEM

      Models              Runtimes             Applications
        │                    │                      │
   PP-OCRv6              PyTorch                 RAG
   GLM-OCR               ONNX                    Agents
   Surya                 TensorRT                Search
   DeepSeek-OCR          vLLM                    ETL
   PaddleOCR-VL          Paddle                  Analytics
        │                    │                      │
        └────────────────────┼──────────────────────┘
                             │
                         TEXTLENS
                             │
                Unified OCR Infrastructure
```

------------------------------------------------------------------------

# 93. The Most Important Strategic Decision

Do not try to win by saying:

> "TextLens has more OCR models."

Win by saying:

> **"TextLens lets you use the entire OCR ecosystem without learning the
> entire OCR ecosystem."**

That is a meaningful developer problem.

------------------------------------------------------------------------

# 94. Final Product Definition

TextLens 2.0 should be:

### An OCR SDK

``` python
OCR(...)
```

### An OCR CLI

``` bash
textlens ocr ...
```

### A document pipeline

``` python
ocr.document(...)
```

### A model manager

``` bash
textlens models ...
```

### A benchmarking platform

``` bash
textlens benchmark ...
```

### A local server

``` bash
textlens serve
```

### An agent tool

``` text
TextLens MCP
```

### A production architecture

``` text
API → Queue → Workers → Inference → Results
```

All sharing the same core API.

------------------------------------------------------------------------

# 95. Final Vision

The end state should feel like this:

``` text
                    ┌─────────────────────┐
                    │      TEXTLENS       │
                    │                     │
                    │ Open OCR Runtime    │
                    └──────────┬──────────┘
                               │
                    What do you want?
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
        OCR                 DOCUMENT              API
          │                    │                    │
     Text / image          PDF / layout         Server
     handwriting           tables               REST
     multilingual          formulas              MCP
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                       Smart Router
                               │
                  ┌────────────┼────────────┐
                  │            │            │
                 Edge        Local        Server
                  │            │            │
               ONNX/GPU    CUDA/VLM      vLLM
                  │            │            │
                  └────────────┼────────────┘
                               │
                        Model Registry
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
          PP-OCR             GLM-OCR        DeepSeek-OCR
          Surya              Paddle-VL      other models
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                       Unified Result
                               │
           ┌───────────┬───────┼────────┬──────────┐
           │           │       │        │          │
          TXT         JSON    MD      HTML       RAG
           │           │       │        │          │
           └───────────┴───────┼────────┴──────────┘
                               │
                       Production Layer
                               │
             Cache · Batch · Queue · Metrics
             Security · Scaling · Provenance
```

**The objective is not to build another OCR model.**

The objective is to build the **open-source infrastructure layer that
makes OCR models practical**.

------------------------------------------------------------------------

# 96. Recommended First Milestone

Do not immediately implement everything in this document.

The first TextLens 2.0 milestone should be:

``` text
MVP-2.0
│
├── pip install textlens
├── OCR(...)
├── unified Result
├── model registry
├── hardware detection
├── doctor
├── PDF classification
├── selective OCR
├── 3–5 high-quality model adapters
├── profiles: edge / fast / balanced / accurate
├── caching
├── batch processing
├── CLI
└── excellent documentation
```

Once this is stable:

``` text
MVP-2.1
├── tables
├── layout
├── structured extraction
├── provenance
├── benchmark
└── visual inspector
```

Then:

``` text
MVP-2.2
├── server
├── async jobs
├── metrics
├── Docker
├── MCP
└── OpenAI-compatible API
```

Then:

``` text
MVP-2.3+
├── vLLM
├── Kubernetes
├── Redis
├── KEDA
├── autoscaling
└── production reference deployments
```

This phased approach prevents TextLens from becoming an enormous,
fragile dependency graph.

------------------------------------------------------------------------

# 97. Design Principle to Keep Throughout Development

> **Simple on the outside. Sophisticated on the inside.**

A beginner should be able to write:

``` python
from textlens import OCR

print(OCR()("image.png").text)
```

while a production engineer should be able to build:

``` text
API Gateway
    ↓
Queue
    ↓
TextLens Router
    ↓
Layout Workers
    ↓
Dynamic Batching
    ↓
vLLM
    ↓
GPU Cluster
    ↓
Result Store
    ↓
MCP / RAG / Application
```

without replacing TextLens.

That is the real goal of TextLens 2.0.

------------------------------------------------------------------------

## Reference Projects

### Firecrawl PDF Inspector

https://github.com/firecrawl/pdf-inspector

Useful ideas to study:

-   fast PDF classification
-   page-level OCR routing
-   native extraction before OCR
-   layout-aware extraction
-   position-aware results
-   Markdown generation
-   selective OCR
-   lightweight defaults
-   multiple language bindings
-   reproducible benchmarking

The current repository explicitly describes a lightweight native path
that avoids OCR when a PDF can be extracted directly, with selective
PP-OCRv6 OCR for pages that need it.

### Neural Maze Production OCR Course

https://github.com/neural-maze/production-ocr-course

Useful ideas to study:

-   production OCR architecture
-   SLM-powered document understanding
-   layout-first processing
-   vLLM
-   continuous batching
-   PagedAttention
-   MTP
-   Redis decoupling
-   Rust ingestion
-   asynchronous workers
-   `/dev/shm` handoff
-   KEDA autoscaling
-   Kubernetes GPU node pools
-   API security
-   MCP

The course explicitly positions itself around the gap between "call an
OCR API" and actually operating OCR as a production system.

------------------------------------------------------------------------

# 98. TextLens 2.0 North Star

``` text
               TEXTLENS

       "OCR, without the OCR complexity."

        Any Document
             ↓
        Any Hardware
             ↓
        Any OCR Model
             ↓
        Any Deployment
             ↓
        One Developer API
```

The framework should make the **right thing the easiest thing**.
