"""
textlens.serving.mcp
────────────────────
TextLens as a Model Context Protocol (MCP) server for AI agents.

    pip install "textlens-ocr[mcp]"
    textlens mcp --workspace ~/documents              # stdio (Claude Code, IDEs)
    textlens mcp --transport sse --port 8765          # network transport

Claude Code configuration (``.mcp.json``)::

    {"mcpServers": {"textlens": {"command": "textlens", "args": ["mcp", "--workspace", "."]}}}

Tools
    ocr_document      text / Markdown for a file (selective OCR, routed models)
    inspect_document  which pages are native vs scanned (no OCR, fast)
    get_page          one page's text with block boxes (for citations)
    search_document   keyword search returning page + bbox citations
    extract_tables    tables as Markdown with page numbers
    extract_fields    schema-driven JSON extraction

Security: every path is resolved against ``--workspace``; anything outside
it is refused.  Documents never leave the machine unless the configured
profile routes to a remote endpoint you set up.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_server(workspace: str = ".", host: str = "127.0.0.1", port: int = 8765, ocr: Any = None) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        from textlens.errors import BackendUnavailableError

        raise BackendUnavailableError("The MCP server needs the 'mcp' package.", hint='pip install "textlens-ocr[mcp]"') from exc
    from textlens.core.engine import OCR
    from textlens.inputs.source import open_source

    root = Path(workspace).expanduser().resolve()
    engine = ocr or OCR()
    server = FastMCP(
        name="textlens",
        instructions=(
            "Local OCR and document intelligence. Paths are relative to the workspace. "
            "Prefer inspect_document first on large PDFs; cite page numbers from get_page/search_document."
        ),
        host=host,
        port=port,
    )

    def src(path: str) -> Any:
        return open_source(path, allowed_root=root, allow_urls=False)

    def pages_arg(pages: Optional[str]) -> Optional[List[int]]:
        from textlens.cli.output import parse_pages

        return parse_pages(pages)

    @server.tool()
    def ocr_document(path: str, pages: Optional[str] = None, format: str = "markdown") -> str:
        """Read a PDF, image, DOCX or PPTX and return its text.

        Args:
            path: File path relative to the workspace.
            pages: Optional page selection like "1-3,7".
            format: "markdown" (default, keeps headings/tables), "text" or "json".
        """
        result = engine(src(path), pages=pages_arg(pages))
        if format == "json":
            return result.to_json()
        if format == "text":
            return result.text
        return result.to_markdown(page_markers=True)

    @server.tool()
    def inspect_document(path: str) -> Dict[str, Any]:
        """Classify a document without OCR: page count, which pages are scanned, and why."""
        report = engine.inspect(src(path))
        if hasattr(report, "pages"):
            return {
                "pdf_type": report.pdf_type,
                "page_count": report.page_count,
                "pages_needing_ocr": report.pages_needing_ocr,
                "reasons_by_page": {str(k): v for k, v in report.reasons_by_page.items()},
                "metadata": report.metadata,
            }
        return report.to_dict() if hasattr(report, "to_dict") else dict(report)

    @server.tool()
    def get_page(path: str, page: int) -> Dict[str, Any]:
        """Return one page (1-indexed) with blocks, types and bounding boxes for citations."""
        result = engine(src(path), pages=[page])
        if not result.pages:
            return {"error": f"page {page} not found"}
        p = result.pages[0]
        return {
            "page": p.number,
            "unit": p.unit,
            "size": [p.width, p.height],
            "source": p.provenance.source,
            "model": p.provenance.model,
            "confidence": p.confidence,
            "blocks": [{"type": b.type, "text": b.text, "bbox": list(b.bbox) if b.bbox else None} for b in p.ordered_blocks()],
        }

    @server.tool()
    def search_document(path: str, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        """Keyword search over a document; returns matching lines with page and bbox."""
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        result = engine(src(path))
        hits = []
        for p in result.pages:
            for ln in p.lines:
                low = ln.text.lower()
                score = sum(low.count(t) for t in terms)
                if score:
                    hits.append({"page": p.number, "text": ln.text, "bbox": list(ln.bbox) if ln.bbox else None, "score": score})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:max_results]

    @server.tool()
    def extract_tables(path: str, pages: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract tables as Markdown, each with its page number."""
        result = engine(src(path), pages=pages_arg(pages), task="table")
        return [{"page": t.page, "markdown": t.to_markdown(), "rows": t.n_rows, "cols": t.n_cols} for t in result.tables]

    @server.tool()
    def extract_fields(path: str, fields: str) -> Dict[str, Any]:
        """Extract named fields to JSON.

        Args:
            path: File path relative to the workspace.
            fields: Comma-separated names (e.g. "invoice_number,date,total") or a JSON schema mapping.
        """
        schema: Any = json.loads(fields) if fields.strip().startswith("{") else [f.strip() for f in fields.split(",") if f.strip()]
        return engine.extract(src(path), schema=schema).to_dict()

    return server


def run_mcp(transport: str = "stdio", workspace: str = ".", host: str = "127.0.0.1", port: int = 8765) -> None:
    server = build_server(workspace=workspace, host=host, port=port)
    server.run(transport=transport)
