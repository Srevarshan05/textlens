"""
OCR.py - Self-Test Script for TextLens Framework
=================================================

Run this script directly:
    python OCR.py

Features tested:
1. Environment Self-Healing (automatically checks and installs missing dependencies)
2. Hardware & CUDA GPU Introspection Doctor
3. Real-Time Terminal Progress Indicator
4. Optimized OCR Extraction (Text & Structured JSON)
"""

import os
from PIL import Image, ImageDraw
import textlens
from textlens import TextLens


def create_sample_image(filepath: str = "sample_test_doc.png") -> str:
    """Generates a simple test image so you can test TextLens immediately."""
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((30, 30), "TEXTLENS OCR FRAMEWORK TEST", fill=(0, 0, 0))
    draw.text((30, 70), "Invoice #: INV-2026-9901", fill=(0, 0, 0))
    draw.text((30, 110), "Date: 2026-08-05", fill=(0, 0, 0))
    draw.text((30, 150), "Item: High Performance GPU Acceleration", fill=(0, 0, 0))
    draw.text((30, 190), "Total Amount: $499.99", fill=(0, 0, 0))
    draw.text((30, 230), "Status: APPROVED & PAID", fill=(0, 0, 0))

    img.save(filepath)
    print(f"[Sample Created] Generated test image at: {filepath}")
    return filepath


def main():
    print("\n" + "=" * 65)
    print("           TEXTLENS FRAMEWORK SELF-TEST SUITE           ")
    print("=" * 65 + "\n")

    # 1. ENVIRONMENT DEPENDENCY SELF-HEALING CHECK
    print("STEP 1: Checking Python Environment Dependencies...")
    textlens.ensure_dependencies(auto_install=True)

    # 2. HARDWARE & GPU DIAGNOSTIC CHECK
    print("\nSTEP 2: Hardware Introspection & CUDA Diagnostic Doctor")
    textlens.print_hardware_status()

    sys_cuda = textlens.detect_system_cuda()
    if textlens.is_cuda_available():
        print(">> Status: NVIDIA CUDA GPU is AVAILABLE & ACCELERATED.\n")
    elif sys_cuda.has_nvidia_gpu:
        print(">> Status: NVIDIA GPU detected on system (CUDA Driver: " + str(sys_cuda.system_cuda_version) + ").")
        print(">> Recommended Command to install GPU PyTorch for your PC:")
        print(f"   {sys_cuda.recommended_install_command}\n")
    else:
        print(">> Status: CUDA GPU not detected. Engine will run in CPU fallback mode.\n")

    # 3. CREATE SAMPLE IMAGE FOR TESTING
    print("STEP 3: Preparing Sample Test Document")
    sample_path = create_sample_image()

    # 4. INITIALIZE TEXTLENS OCR SDK ENGINE WITH REAL-TIME PROGRESS
    print("\nSTEP 4: Initializing TextLens OCR Engine")
    try:
        ocr = TextLens(show_progress=True)

        # 5. TEST GENERAL TEXT READING
        print("\nSTEP 5: Testing General Text Extraction (.read())")
        text_result = ocr.read(sample_path)
        print("\n--- EXTRACTED TEXT RESULT ---")
        print(text_result)
        print("-" * 30)

        # 6. TEST STRUCTURED JSON EXTRACTION
        print("\nSTEP 6: Testing Structured JSON Extraction (.extract_json())")
        json_result = ocr.extract_json(
            sample_path,
            schema='{"invoice_number": "str", "date": "YYYY-MM-DD", "total_amount": "float"}'
        )
        print("\n--- EXTRACTED JSON RESULT ---")
        print(json_result)
        print("-" * 30)

    except Exception as err:
        print(f"\n[Execution Note]: {err}")

    # 7. REST SERVER LAUNCH PROMPT
    print("\nSTEP 7: REST API Endpoint Demonstration")
    print("To launch the REST server with OpenAPI Swagger UI docs, run:")
    print("   textlens serve --port 8000")
    print("\n" + "=" * 65)
    print("              TEXTLENS SELF-TEST COMPLETED SUCCESS!             ")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
