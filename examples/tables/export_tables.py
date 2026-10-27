"""Export every table in a document to CSV, with page provenance.

Native PDF tables with ruling lines are reconstructed without OCR; scanned
tables need a table-capable model (profile "balanced"/"document" with a VLM).

    python examples/tables/export_tables.py statement.pdf out/
"""

import sys
from pathlib import Path

from textlens import OCR

src, out = sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "tables")
out.mkdir(parents=True, exist_ok=True)
result = OCR(profile="document")(src)
for i, table in enumerate(result.tables, start=1):
    path = out / f"page{table.page}-table{i}.csv"
    path.write_text(table.to_csv(), encoding="utf-8")
    print(f"{path}: {table.n_rows}×{table.n_cols} from {table.source} (bbox {table.bbox})")
    print(table.to_markdown())
if not result.tables:
    print("No tables found.")
