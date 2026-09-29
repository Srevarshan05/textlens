"""
textlens.core.quality
─────────────────────
Detect text that *looks* extracted but is actually garbage, so the pipeline
can send the page to OCR instead of returning mojibake.

PDF text layers fail in characteristic ways: CID fonts without a ToUnicode
map, broken CMaps that shift every letter, Type3 glyph fonts, OCR layers
from old scanners.  The detectors below catch those failure modes cheaply.

Attribution
    The detection classes and several calibrated thresholds follow Firecrawl
    pdf-inspector's ``text_quality.rs`` (MIT License, © 2026 Firecrawl,
    https://github.com/firecrawl/pdf-inspector).  This is an independent
    Python implementation; see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List

# English letter frequencies (percent, a–z): a natural-language reference.
# Latin-script languages score ≥ 0.8 cosine similarity against it, while
# substitution-cipher garbling (broken ToUnicode maps) scores ~0.5.
_ENGLISH_FREQ = [
    8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.8, 4.0, 2.4,
    6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0, 2.4, 0.15, 2.0, 0.07,
]
_ENGLISH_NORM = math.sqrt(sum(f * f for f in _ENGLISH_FREQ))
_ENGLISH_SORTED = sorted(_ENGLISH_FREQ, reverse=True)
_ENGLISH_SORTED_NORM = math.sqrt(sum(f * f for f in _ENGLISH_SORTED))

REASON_GARBLED = "suspected_garbled_text"


def _is_private_use(ch: str) -> bool:
    cp = ord(ch)
    return 0xE000 <= cp <= 0xF8FF or 0xF0000 <= cp <= 0xFFFFD or 0x100000 <= cp <= 0x10FFFD


def _is_c1(ch: str) -> bool:
    return 0x80 <= ord(ch) <= 0x9F


@dataclass
class QualityReport:
    """Result of :func:`assess_text`."""

    chars: int = 0
    alnum_ratio: float = 1.0
    replacement_chars: int = 0
    garbled: bool = False
    signals: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.garbled


def has_dollar_as_space(text: str) -> bool:
    """``Word$Word$Word`` — ``$`` used as a separator by a broken CMap."""
    total = text.count("$")
    if total <= 10:
        return False
    letter_dollar_letter = len(re.findall(r"(?<=[A-Za-z])\$(?=[A-Za-z])", text))
    return letter_dollar_letter > 20 or letter_dollar_letter * 2 > total


def is_symbol_soup(text: str) -> bool:
    """Predominantly non-alphanumeric text (≥ 50 scored chars, < 50% alnum).

    Dot/underscore leaders (``.....``) and Markdown syntax are ignored so
    tables of contents and pipe tables are not mistaken for garbage.
    """
    stripped = re.sub(r"([._·])\1{2,}", "", text)
    alnum = non_alnum = 0
    for ch in stripped:
        if ch.isspace() or ch in "#*|-":
            continue
        if ch.isalnum():
            alnum += 1
        else:
            non_alnum += 1
    total = alnum + non_alnum
    return total >= 50 and alnum * 2 < total


def is_cid_garbage(text: str) -> bool:
    """Failed CID→Unicode mapping: C1 controls or Latin-1 mojibake."""
    if is_symbol_soup(text):
        return True
    chars = [c for c in text if not c.isspace() and c != "·"]
    total = len(chars)
    if total < 5:
        return False
    c1 = sum(1 for c in chars if _is_c1(c))
    if c1 >= 2 and c1 * 20 >= total:
        return True
    high_latin = sum(1 for c in chars if 0xA0 <= ord(c) <= 0xFF)
    ascii_letters = sum(1 for c in chars if c.isascii() and c.isalpha())
    return total >= 20 and high_latin * 5 >= total * 2 and ascii_letters * 3 < total


def has_private_use_run(text: str) -> bool:
    total = private = run = longest = 0
    for ch in text:
        if ch.isspace():
            run = 0
            continue
        total += 1
        if _is_private_use(ch):
            private += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return private > 0 and (longest >= 3 or (total >= 5 and private >= 2 and private * 2 >= total))


def looks_cipher_garbled(text: str) -> bool:
    """Substitution-cipher garbling (e.g. ``Certificate`` → ``8VceZWZTReV``).

    Such text is printable ASCII with word-like tokens, so it defeats the
    other detectors.  Two signatures are used: frequent lower→upper case
    flips inside words, or a letter histogram whose *shape* matches natural
    language while the letters sit in the wrong *positions*.
    """
    counts = [0] * 26
    letters = vowels = latin_ext = non_latin = bigrams = case_shifts = 0
    prev = ""
    for ch in text:
        if ch.isascii() and ch.isalpha():
            lo = ch.lower()
            counts[ord(lo) - 97] += 1
            letters += 1
            if lo in "aeiou":
                vowels += 1
            if prev:
                bigrams += 1
                if prev.islower() and ch.isupper():
                    case_shifts += 1
            prev = ch
        else:
            if ch.isalpha():
                if 0xC0 <= ord(ch) <= 0x24F or 0x1E00 <= ord(ch) <= 0x1EFF:
                    latin_ext += 1
                else:
                    non_latin += 1
            prev = ""
    if letters < 200 or non_latin > letters + latin_ext:
        return False
    if vowels / letters > 0.30:
        return False
    if bigrams >= 100 and case_shifts >= bigrams * 0.10:
        return True
    probs = [c / letters for c in counts]
    norm = math.sqrt(sum(p * p for p in probs)) or 1.0
    cosine = sum(p * f for p, f in zip(probs, _ENGLISH_FREQ)) / (norm * _ENGLISH_NORM)
    sorted_probs = sorted(probs, reverse=True)
    shape = sum(p * f for p, f in zip(sorted_probs, _ENGLISH_SORTED)) / (norm * _ENGLISH_SORTED_NORM)
    return cosine < 0.60 and shape >= 0.90


def replacement_needs_ocr(text: str) -> bool:
    """U+FFFD evidence strong enough to distrust a page.

    Short broken layers are condemned by a short run; text-heavy pages need
    density, so a few undecodable math glyphs do not force full-page OCR.
    """
    chars = sum(1 for c in text if not c.isspace())
    repl = text.count("�")
    if not repl or not chars:
        return False
    longest = max((len(m) for m in re.findall("�+", text)), default=0)
    if chars <= 80 and longest >= 2:
        return True
    density_bps = repl * 10_000 // chars
    spans = len(re.findall("�+", text))
    return (
        (repl >= 12 and density_bps >= 500)
        or (spans >= 3 and density_bps >= 250)
        or (longest >= 8 and density_bps >= 250)
    )


def assess_text(text: str) -> QualityReport:
    """Run every detector and summarise the verdict."""
    report = QualityReport()
    nonspace = [c for c in text if not c.isspace()]
    report.chars = len(nonspace)
    if report.chars:
        report.alnum_ratio = sum(1 for c in nonspace if c.isalnum()) / report.chars
    report.replacement_chars = text.count("�")
    if report.chars == 0:
        return report
    checks = (
        ("replacement_characters", replacement_needs_ocr),
        ("private_use_run", has_private_use_run),
        ("dollar_as_space", has_dollar_as_space),
        ("cid_garbage", is_cid_garbage),
        ("cipher_garbled", looks_cipher_garbled),
    )
    for name, check in checks:
        if check(text):
            report.signals.append(name)
    report.garbled = bool(report.signals)
    return report
