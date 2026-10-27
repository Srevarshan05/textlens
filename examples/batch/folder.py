"""Process a folder: one shared model, failures isolated, Markdown written per file.

    python examples/batch/folder.py ./inbox ./out
"""

import sys
from pathlib import Path

from textlens import OCR

src, out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "out")
out.mkdir(parents=True, exist_ok=True)


def on_result(item, res):
    name = Path(str(item)).stem
    if isinstance(res, Exception):
        print(f"✗ {name}: {res}")
        return
    (out / f"{name}.md").write_text(res.to_markdown(page_markers=True), encoding="utf-8")
    print(f"✓ {name}: {res.page_count} page(s) {res.provenance.routing['pages_by_source']}")


OCR(profile="fast", cache="disk").batch(src, workers=4, on_result=on_result)

# The CLI equivalent (with live dashboard and resume):
#   textlens batch ./inbox -o out --format markdown --resume
