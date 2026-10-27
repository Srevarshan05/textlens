"""Route hard pages to a document VLM hosted by vLLM (or any OpenAI-compatible server).

    vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code      # on a GPU host
    python examples/vllm/served_model.py report.pdf http://gpu-host:8000/v1

TextLens still inspects the PDF, extracts native pages locally, and only
sends pages that need OCR to the server — concurrently, so vLLM batches them.
"""

import sys

from textlens import OCR

path, endpoint = sys.argv[1], sys.argv[2]

# Pin the served model…
ocr = OCR(model="paddleocr-vl", endpoint=endpoint)
# …or serve any Hugging Face model the server hosts:
# ocr = OCR(model="deepseek-ai/DeepSeek-OCR-2", backend="openai", endpoint=endpoint)
# …or let the router prefer the server for every OCR page:
# import os; os.environ["TEXTLENS_OPENAI_BASE_URL"] = endpoint; ocr = OCR(profile="server")

result = ocr.document(path)
print(result.to_markdown(page_markers=True)[:2000])
for page in result.pages:
    print(page.number, page.provenance.source, page.provenance.model, page.provenance.timings_ms)
