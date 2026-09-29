"""The smallest useful TextLens program.

    python examples/basic/quickstart.py path/to/file.pdf
"""

import sys

from textlens import OCR

source = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/test-image-ocr.png"
result = OCR()(source)

print(result.text)
print()
print(f"{result.page_count} page(s), confidence {result.confidence}, profile {result.provenance.profile}")
for page in result.pages:
    p = page.provenance
    print(f"  page {page.number}: {p.source:<6} model={p.model or '-'} reasons={p.reasons or '-'}")
