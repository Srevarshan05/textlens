"""
TextLens REST Client Example
============================
Demonstrates querying a running TextLens REST API server endpoint using Python `requests`.
"""

import requests

SERVER_URL = "http://localhost:8000/api/v1/ocr"

def test_url_ocr():
    print(f"Sending URL OCR request to {SERVER_URL} ...")
    payload = {
        "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
        "prompt": "Text Recognition:"
    }
    response = requests.post("http://localhost:8000/api/v1/ocr/json-payload", json=payload)
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print("Device Used:", data.get("device_used"))
        print("Execution Time:", data.get("execution_time_seconds"), "seconds")
        print("Extracted Text:\n", data.get("text"))
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_url_ocr()
