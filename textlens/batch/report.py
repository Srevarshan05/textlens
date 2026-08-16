"""
textlens.batch.report
──────────────────────
Pure-Python PDF Report Generator for BatchOCR.

Generates a standalone, print-ready PDF binary document containing:
- Execution summary & timings
- Model & job configuration
- Hardware telemetry snapshot
- Per-file task processing breakdown table

Zero external dependencies required.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from textlens.batch.types import BatchTask, JobMetrics


class SimplePDFReport:
    """Generates standard PDF 1.4 binary documents for BatchOCR job reports."""

    def __init__(self, metrics: JobMetrics, tasks: List[BatchTask]) -> None:
        self.metrics = metrics
        self.tasks = tasks

    def generate(self) -> bytes:
        """Construct and return PDF 1.4 document bytes."""
        m = self.metrics
        tasks = self.tasks

        # Calculated statistics
        total = m.total_files
        processed = m.processed_files
        failed = m.failed_files
        duration = f"{m.elapsed_time_sec:.1f}s" if m.elapsed_time_sec < 60 else f"{int(m.elapsed_time_sec//60)}m {int(m.elapsed_time_sec%60)}s"
        succ_rate = f"{(processed / total * 100):.1f}%" if total > 0 else "100%"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Build PDF stream text
        stream_cmds = []

        # Color palette
        # Green: 0.44 0.87 0.50 (RGB 112, 223, 127)
        # Dark Bg: 0.05 0.08 0.05
        # Text Dark: 0.1 0.15 0.1

        # Header Box
        stream_cmds.append("0.05 0.08 0.05 rg 0 740 595 102 re f")
        stream_cmds.append("0.44 0.87 0.50 rg 0 738 595 4 re f")
        
        # Header Text
        stream_cmds.append("BT /F2 20 Tf 1 1 1 rg 36 790 Td (TextLens BatchOCR Report) Tj ET")
        stream_cmds.append("BT /F1 10 Tf 0.7 0.8 0.7 rg 36 768 Td (Framework Execution & Document Processing Summary) Tj ET")
        stream_cmds.append(f"BT /F1 9 Tf 0.7 0.8 0.7 rg 420 790 Td (Generated: {self._sanitize(timestamp)}) Tj ET")

        # Section 1: Job Summary
        stream_cmds.append("0.1 0.15 0.1 rg")
        stream_cmds.append("BT /F2 13 Tf 36 710 Td (1. Execution Summary & Configuration) Tj ET")
        stream_cmds.append("0.44 0.87 0.50 RG 2 w 36 702 m 559 702 l S")

        # Config Grid Cards
        items = [
            ("Model ID", m.model_id),
            ("Total Duration", duration),
            ("Processed Files", f"{processed} / {total}"),
            ("Success Rate", succ_rate),
            ("Parallel Workers", f"{m.target_workers} worker(s)"),
            ("Output Format", m.output_format.upper()),
            ("Processing Speed", f"{m.processing_speed_fps:.2f} files/sec"),
            ("GPU Device", m.gpu_name or "NVIDIA CUDA GPU"),
        ]

        y_card = 675
        for i, (lbl, val) in enumerate(items):
            col = i % 2
            row = i // 2
            x = 36 + col * 265
            y = y_card - (row * 32)

            stream_cmds.append(f"0.96 0.98 0.96 rg {x} {y} 250 26 re f")
            stream_cmds.append(f"0.85 0.9 0.85 RG 0.5 w {x} {y} 250 26 re S")
            stream_cmds.append(f"BT /F2 8 Tf 0.4 0.5 0.4 rg {x+8} {y+15} Td ({self._sanitize(lbl)}) Tj ET")
            stream_cmds.append(f"BT /F2 10 Tf 0.1 0.2 0.1 rg {x+8} {y+4} Td ({self._sanitize(str(val))}) Tj ET")

        # Section 2: Document Breakdown Table
        y_table_start = 525
        stream_cmds.append(f"BT /F2 13 Tf 0.1 0.15 0.1 rg 36 {y_table_start} Td (2. Document Processing Breakdown) Tj ET")
        stream_cmds.append(f"0.44 0.87 0.50 RG 2 w 36 {y_table_start-8} m 559 {y_table_start-8} l S")

        # Table Header
        y_th = y_table_start - 30
        stream_cmds.append(f"0.08 0.12 0.08 rg 36 {y_th} 523 20 re f")
        stream_cmds.append(f"BT /F2 9 Tf 1 1 1 rg 42 {y_th+6} Td (#) Tj ET")
        stream_cmds.append(f"BT /F2 9 Tf 1 1 1 rg 65 {y_th+6} Td (File Name) Tj ET")
        stream_cmds.append(f"BT /F2 9 Tf 1 1 1 rg 240 {y_th+6} Td (Status) Tj ET")
        stream_cmds.append(f"BT /F2 9 Tf 1 1 1 rg 310 {y_th+6} Td (Duration) Tj ET")
        stream_cmds.append(f"BT /F2 9 Tf 1 1 1 rg 370 {y_th+6} Td (Pages) Tj ET")
        stream_cmds.append(f"BT /F2 9 Tf 1 1 1 rg 430 {y_th+6} Td (Output File) Tj ET")

        # Table Rows
        y_row = y_th - 20
        max_rows_per_page = 22
        for idx, t in enumerate(tasks[:max_rows_per_page], start=1):
            bg_color = "0.98 0.99 0.98" if idx % 2 == 1 else "0.93 0.96 0.93"
            stream_cmds.append(f"{bg_color} rg 36 {y_row} 523 18 re f")
            stream_cmds.append(f"0.85 0.9 0.85 RG 0.5 w 36 {y_row} 523 18 re S")

            file_name = self._truncate(t.source_path.name, 28)
            dur_str = f"{t.duration_sec:.2f}s" if t.duration_sec > 0 else "-"
            out_str = self._truncate(t.output_path.name if t.output_path else "-", 18)
            status_str = t.status.value

            stream_cmds.append(f"BT /F1 8 Tf 0.2 0.3 0.2 rg 42 {y_row+5} Td ({idx}) Tj ET")
            stream_cmds.append(f"BT /F2 8 Tf 0.1 0.1 0.1 rg 65 {y_row+5} Td ({self._sanitize(file_name)}) Tj ET")
            
            # Status color badge text
            st_color = "0.1 0.6 0.2" if status_str == "COMPLETED" else "0.8 0.1 0.1" if status_str == "FAILED" else "0.7 0.5 0.0"
            stream_cmds.append(f"BT /F2 8 Tf {st_color} rg 240 {y_row+5} Td ({status_str}) Tj ET")

            stream_cmds.append(f"BT /F1 8 Tf 0.2 0.3 0.2 rg 310 {y_row+5} Td ({dur_str}) Tj ET")
            stream_cmds.append(f"BT /F1 8 Tf 0.2 0.3 0.2 rg 370 {y_row+5} Td ({t.page_count or 1}) Tj ET")
            stream_cmds.append(f"BT /F1 8 Tf 0.3 0.4 0.3 rg 430 {y_row+5} Td ({self._sanitize(out_str)}) Tj ET")

            y_row -= 20

        # Footer
        stream_cmds.append("BT /F1 8 Tf 0.5 0.6 0.5 rg 36 30 Td (TextLens Framework · Local First Document AI · MIT License) Tj ET")

        stream_content = "\n".join(stream_cmds).encode("latin1", "replace")

        # Construct PDF Objects
        pdf = bytearray()
        pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets = []

        def add_obj(obj_str: str, binary_data: bytes = b"") -> int:
            offsets.append(len(pdf))
            obj_num = len(offsets)
            pdf.extend(f"{obj_num} 0 obj\n".encode("latin1"))
            pdf.extend(obj_str.encode("latin1"))
            if binary_data:
                pdf.extend(binary_data)
            pdf.extend(b"\nendobj\n")
            return obj_num

        # Obj 1: Catalog
        add_obj("<< /Type /Catalog /Pages 2 0 R >>")
        # Obj 2: Pages
        add_obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        # Obj 3: Page
        add_obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources 4 0 R /Contents 5 0 R >>")
        # Obj 4: Resources (Fonts)
        add_obj("<< /Font << /F1 6 0 R /F2 7 0 R >> >>")
        # Obj 5: Contents
        add_obj(f"<< /Length {len(stream_content)} >>\nstream\n", stream_content + b"\nendstream")
        # Obj 6: Font Helvetica
        add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        # Obj 7: Font Helvetica-Bold
        add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        # Cross-reference table
        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(offsets)+1}\n0000000000 65535 f \n".encode("latin1"))
        for off in offsets:
            pdf.extend(f"{off:010d} 00000 n \n".encode("latin1"))

        # Trailer
        pdf.extend(f"trailer\n<< /Size {len(offsets)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1"))

        return bytes(pdf)

    def _sanitize(self, s: str) -> str:
        """Sanitize text string for PDF Latin-1 font literal encoding."""
        s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return "".join(c if ord(c) < 128 else "?" for c in s)

    def _truncate(self, s: str, max_len: int) -> str:
        return (s[:max_len-2] + "..") if len(s) > max_len else s
