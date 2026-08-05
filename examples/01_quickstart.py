"""
TextLens Quickstart Example
===========================
Demonstrates simple text recognition from an image using auto-detected GPU/CPU.
"""

from textlens import TextLens, print_hardware_status

def main():
    # 1. Inspect hardware status
    print_hardware_status()

    # 2. Initialize TextLens client (auto-detects GPU CUDA or CPU)
    ocr = TextLens()

    # 3. Sample HuggingFace test image URL
    sample_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    print(f"Running OCR on sample URL: {sample_url} ...")
    result = ocr.read(sample_url, prompt="Text Recognition:")

    print("\n" + "=" * 50)
    print("OCR RESULT:")
    print("=" * 50)
    print(result)
    print("=" * 50)

if __name__ == "__main__":
    main()
