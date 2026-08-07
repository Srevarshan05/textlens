from textlens import OCR
from textlens.models import ModelManager
import torch 

# if torch.cuda.is_available():
#     print("Cuda is available!!")
    
# else:
#     print("Not available")

# # 1. View Official Model Catalog
ModelManager.models()
# # 2. Test SmolVLM on an image
# print("\n=== SmolVLM — Image Test ===")
# ocr = OCR(model="smolvlm")
# text = ocr.read("test-image-ocr.png")
# print(text)

# print("\n=== GLM-OCR — Full PDF Test (All Pages) ===")
# ocr_pdf = OCR(model="glm-ocr")
# text_pdf = ocr_pdf.read("tcs_admit_card mail.pdf", dpi=300)
# print(text_pdf)
