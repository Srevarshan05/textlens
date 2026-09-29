"""Plug your own OCR engine into TextLens routing, validation and results.

The example wraps a trivial "engine" that returns one line per image; replace
`_recognize` with calls to your model.

    python examples/custom_backend/my_engine.py tests/fixtures/test-image-ocr.png
"""

import sys

from textlens import OCR, ModelSpec, register_spec
from textlens.backends.base import OCRBackend, PageOCR
from textlens.core.result import BBox
from textlens.documents.layout import Item


class ToyEngine(OCRBackend):
    thread_safe = True
    backend_name = "toy"

    def _load(self):
        self.device = "cpu"  # load weights here (heavy imports inside _load)

    def _recognize(self, images, options):
        results = []
        for img in images:
            w, h = img.size
            item = Item(text=f"image of {w}x{h} pixels", bbox=BBox(0, 0, w, h), confidence=0.99)
            results.append(PageOCR(width=w, height=h, items=[item]))
        return results


register_spec(ModelSpec(
    id="toy-engine", display_name="Toy engine", family="example", provider="you",
    description="Example adapter", backend="onnx", adapter="__main__:ToyEngine",
    tasks=frozenset({"text"}), boxes=True, confidence=True, cpu=True, cpu_practical=True,
    license="mit", commercial_use="yes",
))

result = OCR(model="toy-engine")(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/test-image-ocr.png")
print(result.text, result.pages[0].provenance.model, result.pages[0].blocks[0].bbox)
