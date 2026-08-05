from textlens import OCR, HardwareDoctor
from textlens.models import ModelManager

# 1. View Officially Supported Model Catalog
ModelManager.models()

# 2. Initialize & Run OCR Inference using HunyuanOCR
ocr = OCR(model="hunyuan-ocr")
text = ocr.read("ocr-test-image-2.jpg")

print("\n--- Extracted Text ---")
print(text)
