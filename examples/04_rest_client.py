"""
TextLens Example 04: Sending HTTP Requests to Served Endpoint
"""

import requests

SERVER_URL = "http://localhost:8000/api/v1/ocr"

def test_file_upload(file_path: str):
    """Send image file upload request to TextLens REST Endpoint."""
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"prompt": "Text Recognition:", "max_new_tokens": "512"}
        response = requests.post(SERVER_URL, files=files, data=data)
    
    print("Status Code:", response.status_code)
    print("JSON Response:", response.json())

def test_url_payload(image_url: str):
    """Send image URL request to TextLens JSON REST Endpoint."""
    endpoint = "http://localhost:8000/api/v1/ocr/json-payload"
    payload = {
        "image_url": image_url,
        "prompt": "Text Recognition:",
        "max_new_tokens": 512
    }
    response = requests.post(endpoint, json=payload)
    print("Status Code:", response.status_code)
    print("JSON Response:", response.json())

if __name__ == "__main__":
    print("Example REST Client script for TextLens.")
    print("Ensure `textlens serve` is running on port 8000 before executing!")
