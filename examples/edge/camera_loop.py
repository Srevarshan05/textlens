"""OCR frames from a camera on an edge device (Raspberry Pi / Jetson).

Frames go straight from numpy to TextLens — no temporary files.

    pip install textlens-ocr opencv-python-headless   # OpenCV only for camera capture
    python examples/edge/camera_loop.py
"""

import time

import cv2  # camera capture only; TextLens itself does not need OpenCV

from textlens import OCR

ocr = OCR(profile="edge", cache="off", threads=3)  # leave a core free on a 4-core Pi
ocr.warmup()
cam = cv2.VideoCapture(0)
try:
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = ocr(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # numpy RGB array
        lines = [ln.text for ln in result.lines if (ln.confidence or 0) > 0.8]
        print(f"{(time.perf_counter() - t0) * 1000:.0f} ms", lines)
finally:
    cam.release()
