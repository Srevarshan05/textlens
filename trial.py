from textlens import OCR
from textlens.models import ModelManager

# 1. View Official Model Catalog
ModelManager.models()

# 2. Test SmolVLM on an image
print("\n=== SmolVLM — Image Test ===")
ocr = OCR(model="smolvlm")
text = ocr.read("test-image-ocr.png")
print(text)

# 3. Test SmolVLM on a PDF (specific page 1)
# Uncomment when you have a PDF file:
# print("\n=== SmolVLM — PDF Test (Page 1) ===")
# ocr_pdf = OCR(model="smolvlm")
# text_pdf = ocr_pdf.read("sample.pdf", page=1, dpi=200)
# print(text_pdf)
