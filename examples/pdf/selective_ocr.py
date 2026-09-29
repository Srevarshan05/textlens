"""Inspect a PDF, then process it page by page with selective OCR.

    python examples/pdf/selective_ocr.py report.pdf
"""

import sys
import time

import textlens
from textlens import OCR

path = sys.argv[1]
ocr = OCR(profile="fast")

# 1. Cheap inspection: which pages need OCR, and why (no rendering, no OCR).
report = ocr.inspect(path)
print(report.summary())
for number, reasons in report.reasons_by_page.items():
    print(f"  page {number}: {', '.join(reasons)}")

# 2. Stream pages: constant memory for large documents.
t0 = time.perf_counter()
for page in ocr.stream(path):
    p = page.provenance
    print(f"page {page.number:>4}  {page.classification:<16} {p.source:<6} {len(page.text):>6} chars  {p.model or ''}")
print(f"done in {time.perf_counter() - t0:.1f}s")

# 3. Lazy document: process only what you touch.
doc = textlens.load(path)
print(doc.page_count, "pages; page 1 starts with:", doc.page(1).text[:80].replace("\n", " "))
