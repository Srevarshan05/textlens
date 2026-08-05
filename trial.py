from textlens import OCR, HardwareDoctor
from textlens.models import ModelManager

# # 1. Run Smart Hardware Doctor & Model Recommendations
# doctor = HardwareDoctor()
# report = doctor.run()
# doctor.print_report(report)

# 2. View Officially Supported Model Catalog
ModelManager.models()

# # 3. Initialize & Run OCR Inference using SmolVLM
# ocr = OCR(model="Florence-2")
# text = ocr.read("test-image-ocr.png")

# print("\n--- Extracted Text ---")
# print(text)
