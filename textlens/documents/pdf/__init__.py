"""PDF subsystem: inspection, native extraction and rendering (pypdfium2)."""

from textlens.documents.pdf.inspector import DocumentInspection, InspectionConfig, PageInspection, inspect_pdf

__all__ = ["DocumentInspection", "InspectionConfig", "PageInspection", "inspect_pdf"]
