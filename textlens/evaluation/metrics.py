"""
textlens.evaluation.metrics
───────────────────────────
Text and table accuracy metrics (pure Python; ``rapidfuzz`` accelerates
edit distance automatically when installed).

* **CER** — character edit distance / reference length
* **WER** — word edit distance / reference word count
* **NED** — edit distance / max(len(ref), len(hyp)), in [0, 1]
* **exact match** — after normalisation
* **table cell accuracy** — share of reference cells reproduced at the same
  (row, col), plus a structure (shape) match.  This is a simple, transparent
  proxy; full TEDS needs tree edit distance and is out of scope here.

Corpus-level CER/WER are *micro* averages (total edits / total reference
units), which weight long documents correctly.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Sequence


def _lev_seq(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def levenshtein(a: Sequence, b: Sequence) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    except ImportError:
        return _lev_seq(a, b)


def normalize_text(text: str, lowercase: bool = False, strip_punctuation: bool = False) -> str:
    """NFKC, unify whitespace/newlines, optionally lowercase and drop punctuation."""
    t = unicodedata.normalize("NFKC", text or "")
    if lowercase:
        t = t.lower()
    if strip_punctuation:
        t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def ned(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    m = max(len(ref), len(hyp))
    return levenshtein(ref, hyp) / m if m else 0.0


def exact_match(reference: str, hypothesis: str, lowercase: bool = False) -> bool:
    return normalize_text(reference, lowercase) == normalize_text(hypothesis, lowercase)


def text_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    ref_w, hyp_w = ref.split(), hyp.split()
    return {
        "char_edits": float(levenshtein(ref, hyp)),
        "ref_chars": float(len(ref)),
        "word_edits": float(levenshtein(ref_w, hyp_w)),
        "ref_words": float(len(ref_w)),
        "cer": cer(reference, hypothesis),
        "wer": wer(reference, hypothesis),
        "ned": ned(reference, hypothesis),
        "exact": 1.0 if ref == hyp else 0.0,
    }


def table_cell_accuracy(reference: List[List[str]], hypothesis: List[List[str]]) -> Dict[str, float]:
    ref_cells = {(r, c): normalize_text(v, True) for r, row in enumerate(reference) for c, v in enumerate(row)}
    hyp_cells = {(r, c): normalize_text(v, True) for r, row in enumerate(hypothesis) for c, v in enumerate(row)}
    if not ref_cells:
        return {"cell_accuracy": 1.0 if not hyp_cells else 0.0, "structure_match": 1.0 if not hyp_cells else 0.0}
    correct = sum(1 for k, v in ref_cells.items() if hyp_cells.get(k) == v)
    shape_ref = (len(reference), max((len(r) for r in reference), default=0))
    shape_hyp = (len(hypothesis), max((len(r) for r in hypothesis), default=0))
    return {"cell_accuracy": correct / len(ref_cells), "structure_match": 1.0 if shape_ref == shape_hyp else 0.0}
