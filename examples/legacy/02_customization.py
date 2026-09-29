"""
TextLens Example 02: Advanced Extraction & Customization
"""

from textlens import TextLens

def main():
    ocr = TextLens()

    # 1. Custom Prompt Instruction
    # custom_text = ocr.read("document.png", prompt="Extract header information and date:", max_new_tokens=256)

    # 2. Markdown Table Extraction
    # table_md = ocr.extract_table("financial_statement.png")
    # print("Markdown Table:\n", table_md)

    # 3. LaTeX Formula Extraction
    # math_latex = ocr.extract_formula("equation.png")
    # print("LaTeX Formula:\n", math_latex)

    # 4. Structured JSON Output
    # json_data = ocr.extract_json(
    #     "invoice.png",
    #     schema='{"invoice_number": "str", "total": "float", "date": "YYYY-MM-DD"}'
    # )
    # print("Structured JSON:\n", json_data)

    # 5. Multi-Page PDF OCR
    # pdf_pages = ocr.read_pdf("sample.pdf", max_pages=3)
    # for p in pdf_pages:
    #     print(f"Page {p['page']} text length: {len(p['text'])} characters")

    # 6. Dynamic Device Switching
    if ocr.hardware.gpu_available:
        ocr.switch_device("cpu")
        ocr.switch_device("cuda")

if __name__ == "__main__":
    main()
