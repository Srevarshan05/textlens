"""Read number plates from images or a video.

    pip install "textlens-ocr[anpr]"
    python examples/anpr/plates.py car.jpg --region in
    python examples/anpr/plates.py traffic.mp4 --region uk      # needs opencv-python-headless
"""

import argparse
from pathlib import Path

from textlens import ANPR

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("--region", default="generic")
parser.add_argument("--profile", default="balanced", choices=["edge", "balanced", "accurate"])
args = parser.parse_args()

anpr = ANPR(profile=args.profile, region=args.region)

if Path(args.source).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
    import cv2

    cap = cv2.VideoCapture(args.source)

    def frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    from PIL import Image

    for result in anpr.stream(Image.fromarray(f) for f in frames()):
        for p in result.plates:
            print(f"frame {result.frame}: track {p.track_id} {p.plate} ({p.confidence:.2f}, valid={p.valid})")
else:
    result = anpr(args.source)
    for p in result.plates:
        print(p.plate, f"{p.confidence:.3f}", "valid" if p.valid else "unverified", p.format or "", p.corrections, p.bbox)
    result.annotate(args.source).save("plates_annotated.jpg")
    print("wrote plates_annotated.jpg")
