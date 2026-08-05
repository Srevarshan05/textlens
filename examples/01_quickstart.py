"""
TextLens Example 01: Quickstart & Device Inspection
"""

import textlens
from textlens import TextLens

def main():
    # 1. Print system hardware capability (CUDA GPU vs CPU)
    textlens.print_hardware_status()

    # 2. Programmatically check CUDA availability
    if textlens.is_cuda_available():
        print("✅ NVIDIA CUDA GPU acceleration is ENABLED.")
    else:
        print("⚠️  Running on CPU fallback mode.")

    # 3. Initialize SDK client
    ocr = TextLens()

    # 4. Perform simple OCR reading
    # text = ocr.read("path/to/invoice.png")
    # print("Extracted Text:", text)

if __name__ == "__main__":
    main()
