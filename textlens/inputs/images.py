"""
textlens.inputs.images
──────────────────────
Image loading and lightweight preprocessing shared by every backend.

Only Pillow and numpy are used, so this works on edge devices without
OpenCV.  Heavier, model-specific preprocessing belongs in the adapters.
"""

from __future__ import annotations

import io
from typing import Any, Iterator, List, Tuple

from textlens.errors import InputError, UnsupportedInputError

# Refuse pathological images (decompression bombs) but allow large scans:
# 20k x 20k is far beyond any sensible page raster.
MAX_IMAGE_PIXELS = 400_000_000


def _pil():
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    return Image, ImageOps


def to_rgb(img: Any) -> Any:
    """Convert any Pillow mode to RGB, flattening transparency onto white."""
    Image, _ = _pil()
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        import numpy as np

        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        scaled = ((arr - lo) / (hi - lo) * 255.0) if hi > lo else arr * 0
        return Image.fromarray(scaled.astype("uint8")).convert("RGB")
    return img.convert("RGB")


def array_to_image(arr: Any) -> Any:
    """Convert a numpy array (HxW or HxWxC, RGB order) to a PIL RGB image."""
    import numpy as np

    Image, _ = _pil()
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        if a.dtype.kind == "f" and a.max() <= 1.0:
            a = (a * 255.0).clip(0, 255)
        a = a.clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a, "L").convert("RGB")
    if a.ndim == 3 and a.shape[2] in (1, 3, 4):
        if a.shape[2] == 1:
            return Image.fromarray(a[:, :, 0], "L").convert("RGB")
        return to_rgb(Image.fromarray(a))
    raise UnsupportedInputError(f"Unsupported array shape {a.shape}; expected HxW or HxWxC.")


def iter_image_frames(data_or_file: Any) -> Iterator[Any]:
    """Yield RGB frames of an image file (multi-page TIFF/GIF yield several)."""
    Image, ImageOps = _pil()
    try:
        img = Image.open(data_or_file if not isinstance(data_or_file, bytes) else io.BytesIO(data_or_file))
    except Exception as exc:
        raise InputError(f"Cannot decode image: {exc}") from exc
    n = getattr(img, "n_frames", 1)
    for i in range(n):
        if n > 1:
            img.seek(i)
        frame = ImageOps.exif_transpose(img) if i == 0 else img.copy()
        yield to_rgb(frame)


def load_image(obj: Any) -> Any:
    """Return the first frame of an image source as an RGB PIL image."""
    for frame in iter_image_frames(obj):
        return frame
    raise InputError("Image contains no frames.")


def limit_size(img: Any, max_side: int) -> Tuple[Any, float]:
    """Downscale so the longest side is ``<= max_side``; return ``(img, scale)``."""
    Image, _ = _pil()
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / float(longest)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), scale


def to_grayscale_array(img: Any, max_side: int = 1024) -> Any:
    """Small grayscale float array (0–1) used by analysis heuristics."""
    import numpy as np

    small, _ = limit_size(img, max_side)
    return np.asarray(small.convert("L"), dtype=np.float32) / 255.0


def frames_of(images: List[Any]) -> List[Any]:
    return [to_rgb(im) for im in images]
