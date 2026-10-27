"""Turn documents into citation-ready chunks for a vector store.

    python examples/rag/chunks.py handbook.pdf
"""

import json
import sys

from textlens import OCR

result = OCR()(sys.argv[1])
chunks = result.to_chunks(max_tokens=400, overlap_tokens=40, strategy="semantic")

records = [
    {
        "id": f"{c.document_id[:12]}-{c.index}",
        "text": c.text,
        "metadata": {
            "source": result.metadata["source_name"],
            "pages": c.pages,
            "section": " > ".join(c.section),
            "boxes": c.bboxes,  # [{"page": n, "bbox": [x0, y0, x1, y1]}] in PDF points
        },
    }
    for c in chunks
]
print(json.dumps(records[:2], indent=2, ensure_ascii=False))
print(f"... {len(records)} chunks")
# store.add(records)  # your vector DB here
