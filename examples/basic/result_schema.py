"""Walk the unified result: pages → blocks → lines → words, with boxes and provenance.

    python examples/basic/result_schema.py tests/fixtures/invoice.png
"""

import sys

from textlens import OCR

result = OCR()(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/invoice.png")

for page in result.pages:
    print(f"Page {page.number}  {page.width:.0f}×{page.height:.0f} {page.unit}  source={page.provenance.source}")
    for block in page.ordered_blocks():
        print(f"  [{block.order}] {block.type:<9} conf={block.confidence} bbox={block.bbox}")
        for line in block.lines[:2]:
            words = ", ".join(f"{w.text}@{tuple(round(v) for v in w.bbox)}" for w in line.words[:4] if w.bbox)
            print(f"        {line.text!r}  words: {words}")

# Exports
open("result.md", "w", encoding="utf-8").write(result.to_markdown(page_markers=True))
open("result.json", "w", encoding="utf-8").write(result.to_json())
print("\nwrote result.md and result.json  ·  fingerprint:", result.provenance.config_hash, result.provenance.document_hash[:12])
