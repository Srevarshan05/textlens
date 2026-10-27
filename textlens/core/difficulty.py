"""
textlens.core.difficulty
────────────────────────
Cheap, model-free page difficulty estimation for routing.

Signals (on a ≤1024 px grayscale copy, a few milliseconds):

* **contrast**      intensity spread; faded scans and photos score low
* **sharpness**     variance of a Laplacian; blur scores low
* **ink density**   share of dark pixels; dense or noisy pages score high
* **ruling lines**  long horizontal + vertical dark runs → likely a table
* **text scale**    tiny text on a small image is harder to read

The combined ``difficulty`` in [0, 1] is a *heuristic* routing hint, not a
quality prediction: < 0.35 simple, 0.35–0.65 moderate, > 0.65 hard.  Use
``textlens benchmark`` to calibrate routing on your own documents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ImageSignals:
    contrast: float
    sharpness: float
    ink_density: float
    h_lines: int
    v_lines: int
    width: int
    height: int
    difficulty: float

    @property
    def likely_table(self) -> bool:
        return self.h_lines >= 3 and self.v_lines >= 2

    @property
    def level(self) -> str:
        return "simple" if self.difficulty < 0.35 else "moderate" if self.difficulty < 0.65 else "hard"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["likely_table"] = self.likely_table
        d["level"] = self.level
        return d


def _line_runs(mask: np.ndarray, axis: int, min_frac: float) -> int:
    """Count rows (axis=1) or columns (axis=0) that are mostly dark."""
    frac = mask.mean(axis=axis)
    hits = frac >= min_frac
    # Collapse adjacent hits (a thick rule is one line).
    return int(np.count_nonzero(hits[1:] & ~hits[:-1]) + (1 if hits.size and hits[0] else 0))


def analyze_image(img: Any) -> ImageSignals:
    from textlens.inputs.images import to_grayscale_array

    g = to_grayscale_array(img, max_side=1024)
    h, w = g.shape
    contrast = float(np.percentile(g, 95) - np.percentile(g, 5))
    lap = (
        -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
        if h > 2 and w > 2
        else np.zeros((1, 1), np.float32)
    )
    sharpness = float(lap.var())
    dark = g < min(0.5, float(np.percentile(g, 50)) - 0.15 * contrast) if contrast > 0.05 else g < 0.5
    ink = float(dark.mean())
    h_lines = _line_runs(dark, axis=1, min_frac=0.45)
    v_lines = _line_runs(dark, axis=0, min_frac=0.35)

    score = 0.0
    score += 0.30 * float(np.clip((0.55 - contrast) / 0.45, 0, 1))  # low contrast
    score += 0.25 * float(np.clip((0.004 - sharpness) / 0.004, 0, 1))  # blur
    score += 0.20 * float(np.clip((ink - 0.12) / 0.25, 0, 1))  # dense / noisy
    score += 0.15 * (1.0 if (h_lines >= 3 and v_lines >= 2) else 0.0)  # tables
    score += 0.10 * float(np.clip((600 - max(img.size)) / 600, 0, 1))  # tiny image
    return ImageSignals(
        contrast=round(contrast, 3),
        sharpness=round(sharpness, 5),
        ink_density=round(ink, 4),
        h_lines=h_lines,
        v_lines=v_lines,
        width=int(img.size[0]),
        height=int(img.size[1]),
        difficulty=round(float(np.clip(score, 0.0, 1.0)), 3),
    )
