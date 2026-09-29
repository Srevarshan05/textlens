# TextLens examples

Run from the repository root after `pip install -e .` (add extras where noted).

| Folder | Example | Needs |
|---|---|---|
| `basic/` | `quickstart.py` — one call, text + provenance · `result_schema.py` — pages, blocks, boxes, exports | core |
| `pdf/` | `selective_ocr.py` — inspect, stream pages, lazy document | core |
| `tables/` | `export_tables.py` — tables to CSV/Markdown | core (VLM for scanned tables) |
| `rag/` | `chunks.py` — citation-ready chunks | core |
| `extraction/` | `invoice_fields.py` — schema → JSON with boxes | core |
| `batch/` | `folder.py` — shared model, isolated failures | core |
| `server/` | `client.py` — sync + async jobs against `textlens serve` | server extra (on the server) |
| `vllm/` | `served_model.py` — route pages to a vLLM-hosted VLM | an OpenAI-compatible endpoint |
| `edge/` | `camera_loop.py` — camera frames on a Pi/Jetson | core + OpenCV for capture |
| `anpr/` | `plates.py` — plates from images or video, tracking | anpr extra |
| `custom_backend/` | `my_engine.py` — plug in your own engine | core |
| `mcp/` | MCP configuration for AI agents | mcp extra |
| `legacy/` | the 0.x examples (still supported) | as noted there |

Deployment examples live in `deploy/` (Docker, Compose, Kubernetes).
Benchmark your own data with `textlens benchmark` (see
`docs/evaluation/benchmarking.md`).
