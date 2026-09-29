# TextLens as an MCP server

```bash
pip install "textlens-ocr[mcp]"
```

## Claude Code / IDE agents (stdio)

`.mcp.json` in your project:

```json
{
  "mcpServers": {
    "textlens": {
      "command": "textlens",
      "args": ["mcp", "--workspace", "."]
    }
  }
}
```

## Network transport

```bash
textlens mcp --transport sse --host 127.0.0.1 --port 8765 --workspace /srv/docs
```

## Tools

| Tool | Returns |
|---|---|
| `ocr_document(path, pages?, format?)` | Markdown (default), text or JSON |
| `inspect_document(path)` | document type, pages needing OCR, reasons |
| `get_page(path, page)` | blocks with types and bounding boxes (for citations) |
| `search_document(path, query)` | matching lines with page + bbox |
| `extract_tables(path, pages?)` | tables as Markdown with page numbers |
| `extract_fields(path, fields)` | schema-driven JSON |

Paths are resolved against `--workspace`; anything outside it is refused.
Documents are processed locally with your saved TextLens profile.
