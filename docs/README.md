# TextLens documentation

TextLens is an OCR **infrastructure layer**: it inspects documents, extracts
native text where it can, routes the rest to the right OCR engine for your
hardware, validates the output and returns one result schema — from a
Raspberry Pi to a GPU server.

## Start here

| If you want to… | Read |
|---|---|
| install and run your first OCR | [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) |
| fix an error | [Troubleshooting](getting-started/troubleshooting.md) |
| upgrade from 0.x | [Migration guide](migration.md) |

## Concepts

- [Architecture](concepts/architecture.md) — the pipeline, and why each stage exists
- [Results](concepts/results.md) — pages, blocks, boxes, confidence, provenance, exports
- [Routing and profiles](concepts/routing.md) — how TextLens picks a model, and how to override it
- [Models](concepts/models.md) — the registry, model cards, licenses, adding your own engine

## Tasks

- [PDFs and selective OCR](tasks/pdf.md) — native vs scanned pages, reason codes, large documents
- [Tables and layout](tasks/tables-layout.md) — reading order, headings, tables
- [Structured extraction](tasks/extraction.md) — schema → JSON
- [RAG](tasks/rag.md) — chunks with citations
- [Number plates (ANPR)](tasks/anpr.md)

## Deployment

- [GPU setup](deployment/gpu.md) · [Edge devices](deployment/edge.md) · [Raspberry Pi](deployment/raspberry-pi.md) · [Jetson](deployment/jetson.md)
- [REST server](deployment/server.md) · [Docker](deployment/docker.md)

## Production

- [Scaling, batching and caching](production/scaling.md)
- [Security and privacy](production/security.md)
- [Observability](production/observability.md)

## Evaluation

- [Benchmarking, profiling and visual debugging](evaluation/benchmarking.md)

## Design notes

- [2.0 audit and migration plan](design/audit-2.0.md)
- [PDF subsystem and Firecrawl pdf-inspector](design/pdf-inspector.md)
- [API reference](API_REFERENCE.md)
