"""
textlens.backends.onnx.dbnet
────────────────────────────
DBNet text-detection post-processing in pure numpy (no OpenCV).

Pipeline: probability map → threshold → connected components (run-length
union-find, 8-connectivity) → mean-probability score per component →
convex hull → minimum-area rectangle (rotating calipers) → unclip
(offset by ``area * ratio / perimeter``) → quadrilateral in image pixels.

Avoiding OpenCV keeps the default install small and sidesteps the common
``opencv-python`` / ``opencv-python-headless`` conflict on edge devices.
Run-length labelling keeps the Python loop proportional to text *runs*, not
pixels, so a 960×960 map post-processes in ~10–60 ms.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _runs(bitmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.zeros((bitmap.shape[0], bitmap.shape[1] + 2), np.int8)
    padded[:, 1:-1] = bitmap
    diff = np.diff(padded, axis=1)
    ys, xs = np.nonzero(diff == 1)
    _, xe = np.nonzero(diff == -1)
    return ys, xs, xe  # run r covers [xs[r], xe[r]) on row ys[r]


def _label(ys: np.ndarray, xs: np.ndarray, xe: np.ndarray) -> np.ndarray:
    n = len(ys)
    parent = list(range(n))  # plain list: scalar access is far faster than ndarray

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    if n == 0:
        return np.arange(0)
    max_y = int(ys.max())
    starts = np.searchsorted(ys, np.arange(max_y + 3))
    for y in range(max_y):
        a0, a1 = int(starts[y]), int(starts[y + 1])
        b0, b1 = int(starts[y + 1]), int(starts[y + 2])
        if a0 == a1 or b0 == b1:
            continue
        prev_end, prev_start = xe[a0:a1], xs[a0:a1]
        for j in range(b0, b1):
            lo = int(np.searchsorted(prev_end, xs[j], "left"))  # prev.end >= start (8-conn)
            hi = int(np.searchsorted(prev_start, xe[j], "right"))  # prev.start <= end
            for i in range(a0 + lo, a0 + hi):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts.astype(np.float64)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], np.float64)


def min_area_rect(hull: np.ndarray) -> Tuple[float, float, float, float, float]:
    """``(cx, cy, w, h, angle)`` of the minimum-area enclosing rectangle."""
    if len(hull) < 3:
        (x0, y0), (x1, y1) = hull.min(0), hull.max(0)
        return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    best = None
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, s], [-s, c]])
        r = hull @ rot.T
        lo, hi = r.min(0), r.max(0)
        area = float((hi - lo).prod())
        if best is None or area < best[0]:
            ctr = ((lo + hi) / 2) @ rot
            best = (area, float(ctr[0]), float(ctr[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(a))
    return best[1:]  # type: ignore[index]


def rect_points(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    ux, uy = np.array([c, s]) * w / 2, np.array([-s, c]) * h / 2
    ctr = np.array([cx, cy])
    return np.array([ctr - ux - uy, ctr + ux - uy, ctr + ux + uy, ctr - ux + uy])


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def boxes_from_probability(
    prob: np.ndarray,
    scale_xy: Tuple[float, float],
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    unclip_ratio: float = 1.5,
    min_size: float = 3.0,
    max_candidates: int = 3000,
) -> List[Tuple[np.ndarray, float]]:
    """Return ``[(quad (4×2, image px), score)]`` from a DB probability map."""
    ys, xs, xe = _runs(prob > thresh)
    if len(ys) == 0:
        return []
    labels = _label(ys, xs, xe)
    csum = np.zeros((prob.shape[0], prob.shape[1] + 1), np.float64)
    csum[:, 1:] = np.cumsum(prob, axis=1)
    run_sum = csum[ys, xe] - csum[ys, xs]
    run_len = (xe - xs).astype(np.float64)
    _, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    scores = np.bincount(inverse, run_sum) / np.bincount(inverse, run_len)
    order = np.argsort(inverse, kind="stable")
    groups = np.split(order, np.cumsum(counts)[:-1])
    sx, sy = scale_xy
    out: List[Tuple[np.ndarray, float]] = []
    for k, idx in enumerate(groups[:max_candidates]):
        if scores[k] < box_thresh or run_len[idx].sum() < 8:
            continue
        pts = np.concatenate(
            [
                np.stack([xs[idx], ys[idx]], 1),
                np.stack([xe[idx], ys[idx]], 1),
                np.stack([xs[idx], ys[idx] + 1], 1),
                np.stack([xe[idx], ys[idx] + 1], 1),
            ]
        ).astype(np.float64)
        cx, cy, w, h, a = min_area_rect(convex_hull(pts))
        if min(w, h) < min_size:
            continue
        dist = w * h * unclip_ratio / (2.0 * (w + h))
        quad = rect_points(cx, cy, w + 2 * dist, h + 2 * dist, a)
        quad[:, 0] *= sx
        quad[:, 1] *= sy
        out.append((order_quad(quad), float(scores[k])))
    return out
