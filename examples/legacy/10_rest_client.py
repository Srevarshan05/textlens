"""Upload an image to the REST API started by 09_rest_server.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with args.image.open("rb") as file_handle:
        response = requests.post(
            f"{args.base_url}/api/v1/ocr",
            files={"file": file_handle},
            data={"prompt": "Text Recognition:", "max_new_tokens": "512"},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    print("Device:", payload["device_used"])
    print("Time (seconds):", payload["execution_time_seconds"])
    print(payload.get("text", payload.get("pages")))


if __name__ == "__main__":
    main()
