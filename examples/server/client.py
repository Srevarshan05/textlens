"""Call a running TextLens server (stdlib only).

    textlens serve --port 8000           # in another terminal (TEXTLENS_API_KEYS=k1 to require a key)
    python examples/server/client.py scan.pdf
"""

import json
import sys
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"
KEY = "k1"


def post_file(path: str, url: str, **fields: str) -> dict:
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path}"\r\n\r\n'.encode()
    body += open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-API-Key": KEY})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"X-API-Key": KEY})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


path = sys.argv[1]
# Synchronous
result = post_file(path, f"{BASE}/ocr")
print(result["text"][:300], "\nconfidence:", result["confidence"])

# Asynchronous job with polling
job = post_file(path, f"{BASE}/ocr", **{"async": "true"})
while (status := get(f"{BASE}/jobs/{job['job_id']}"))["status"] in ("queued", "running"):
    print("progress:", status["progress"])
    time.sleep(0.5)
print("job:", status["status"], "pages:", len(get(f"{BASE}/jobs/{job['job_id']}/result")["pages"]))
