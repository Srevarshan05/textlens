"""
textlens.anpr.validation
────────────────────────
Normalise, validate and *correct* plate readings against region formats.

OCR confuses shapes (``O``/``0``, ``I``/``1``, ``B``/``8``, ``S``/``5``…).
Knowing whether a position expects a letter or a digit resolves most of
these: the validator tries the format templates of the region and applies
the cheapest position-aware substitutions that make the reading valid,
preferring corrections at low-confidence characters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from textlens.anpr.formats import Region

# Letter → digit and digit → letter substitutions that OCR commonly confuses.
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "1", "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "E", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}


@dataclass
class Validation:
    text: str
    valid: bool
    format: Optional[str] = None
    corrections: List[str] = field(default_factory=list)
    score: float = 0.0  # 1.0 = valid without edits; lower with corrections


def normalize(text: str) -> str:
    """Uppercase, strip accents/separators, keep A–Z and 0–9."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]", "", s.upper())


class PlateValidator:
    def __init__(self, region: Region, max_corrections: int = 3) -> None:
        self.region = region
        self.max_corrections = max_corrections
        self._patterns = [(f, f.pattern()) for f in region.formats]

    def validate(self, raw: str, char_confidences: Optional[Sequence[float]] = None) -> Validation:
        text = normalize(raw)
        if not (self.region.min_length <= len(text) <= self.region.max_length):
            return Validation(text, False, score=0.0)
        for fmt, pat in self._patterns:
            if pat.match(text):
                return Validation(text, True, fmt.name, score=1.0)
        best: Optional[Tuple[float, str, str, List[str]]] = None
        confs = list(char_confidences or [])
        for fmt, pat in self._patterns:
            slots = fmt.slots(len(text))
            if not slots:
                continue
            candidate, edits, cost = self._correct(text, slots, confs)
            if candidate is None or len(edits) > self.max_corrections or not pat.match(candidate):
                continue
            score = max(0.1, 1.0 - 0.15 * cost)
            if best is None or score > best[0]:
                best = (score, candidate, fmt.name, edits)
        if best is not None:
            return Validation(best[1], True, best[2], best[3], best[0])
        return Validation(text, False, score=0.0)

    @staticmethod
    def _correct(text: str, slots: List[str], confs: List[float]) -> Tuple[Optional[str], List[str], float]:
        chars = list(text)
        edits: List[str] = []
        cost = 0.0
        for i, (ch, cls) in enumerate(zip(chars, slots)):
            if cls == "D" and not ch.isdigit():
                rep = TO_DIGIT.get(ch)
            elif cls == "L" and not ch.isalpha():
                rep = TO_LETTER.get(ch)
            else:
                continue
            if rep is None:
                return None, edits, cost
            conf = confs[i] if i < len(confs) else 0.5
            cost += 0.5 + conf  # confident characters are expensive to change
            edits.append(f"pos {i + 1}: {ch}→{rep}")
            chars[i] = rep
        return "".join(chars), edits, cost
