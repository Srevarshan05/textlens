"""Extract invoice fields to JSON (local heuristics, or a VLM when available).

    python examples/extraction/invoice_fields.py tests/fixtures/invoice.png
"""

import sys

from textlens import OCR

schema = {"vendor": "string", "invoice_number": "string", "date": "date", "subtotal": "float", "total": "float"}
ext = OCR().extract(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/invoice.png", schema=schema)

print("method:", ext.method)
for name, value in ext.data.items():
    info = ext.fields[name]
    where = f"page {info['page']} {info.get('bbox')}" if info.get("found") else "not found"
    print(f"  {name:<15} {value!r:<25} {where}")
