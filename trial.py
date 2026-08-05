from textlens import OCR, HardwareDoctor
from textlens.models import ModelManager

# 1. View Officially Supported Model Catalog
ModelManager.models()

# 2. Initialize & Run OCR Inference using LightOnOCR
ocr = OCR(model="lighton-ocr")
text = ocr.read("test-image-ocr.png")

print("\n--- Extracted Text ---")
print(text)
