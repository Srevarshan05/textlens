"""
textlens.anpr.formats
─────────────────────
Configurable plate grammars.  Nothing country-specific is hard-coded in the
pipeline: a *region* is a list of :class:`PlateFormat` rules.

A format is a compact template over position classes::

    L  letter        D  digit        A  letter or digit
    X{m,n}  repeat the previous class m..n times (regex-style braces)
    ( )     optional group is not supported — list alternatives instead

e.g. India standard series ``LLDDL{1,3}DDDD`` matches ``TN01AB1234``.

Custom regions load from JSON (or YAML if PyYAML is installed)::

    {"name": "my-country", "formats": [
        {"name": "private", "template": "LLL-DDD", "strip": "-"},
        {"name": "taxi", "regex": "^T[0-9]{5}$"}
    ]}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_CLASS = {"L": "[A-Z]", "D": "[0-9]", "A": "[A-Z0-9]"}


@dataclass(frozen=True)
class PlateFormat:
    name: str
    template: Optional[str] = None  # position-class template (see module doc)
    regex: Optional[str] = None  # alternative: a full regex over the normalised text
    description: str = ""

    def pattern(self) -> "re.Pattern[str]":
        if self.regex:
            return re.compile(self.regex)
        return re.compile("^" + template_to_regex(self.template or "") + "$")

    def slots(self, length: int) -> Optional[List[str]]:
        """Expand the template to per-position classes for a given length."""
        if not self.template:
            return None
        for expansion in expand_template(self.template):
            if len(expansion) == length:
                return expansion
        return None


@dataclass(frozen=True)
class Region:
    name: str
    formats: Sequence[PlateFormat] = field(default_factory=tuple)
    min_length: int = 4
    max_length: int = 10
    description: str = ""


def template_to_regex(template: str) -> str:
    out = []
    i = 0
    while i < len(template):
        ch = template[i]
        if ch in _CLASS:
            out.append(_CLASS[ch])
        elif ch == "{":
            j = template.index("}", i)
            out.append(template[i : j + 1])
            i = j
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


def expand_template(template: str, limit: int = 64) -> List[List[str]]:
    """All concrete slot sequences for a template (bounded)."""
    tokens: List[tuple] = []  # (class, min, max)
    i = 0
    while i < len(template):
        ch = template[i]
        if ch == "{" and tokens:
            j = template.index("}", i)
            body = template[i + 1 : j]
            lo, _, hi = body.partition(",")
            cls = tokens[-1][0]
            tokens[-1] = (cls, int(lo), int(hi or lo))
            i = j + 1
            continue
        if ch in _CLASS:
            tokens.append((ch, 1, 1))
        i += 1
    results: List[List[str]] = [[]]
    for cls, lo, hi in tokens:
        results = [r + [cls] * n for r in results for n in range(lo, hi + 1)]
        if len(results) > limit:
            results = results[:limit]
    return results


# ── built-in regions (extend with a JSON/YAML file) ─────────────────────────
REGIONS: Dict[str, Region] = {
    "generic": Region("generic", (PlateFormat("alnum", regex=r"^[A-Z0-9]{4,10}$", description="Any 4–10 alphanumerics"),),
                      description="No country rules; only normalisation and length checks."),
    "in": Region(
        "in",
        (
            PlateFormat("standard", "LLDDL{1,3}DDDD", description="State + RTO + series + number, e.g. TN01AB1234"),
            PlateFormat("standard-short-rto", "LLDL{1,3}DDDD", description="Single-digit RTO, e.g. DL3CAB1234"),
            PlateFormat("no-series", "LLDDDDDD", description="Older plates without series letters"),
            PlateFormat("bharat", "DDLLDDDDL{1,2}", description="BH series, e.g. 22BH1234AA"),
        ),
        description="India (standard, BH series)",
    ),
    "uk": Region("uk", (PlateFormat("current", "LLDDLLL", description="e.g. AB12CDE"),), description="United Kingdom (2001 format)"),
    "de": Region("de", (PlateFormat("standard", regex=r"^[A-ZÄÖÜ]{1,3}[A-Z]{1,2}[0-9]{1,4}[EH]?$", description="e.g. BAB1234"),), description="Germany"),
    "fr": Region("fr", (PlateFormat("siv", "LLDDDLL", description="e.g. AB123CD"),), description="France (SIV)"),
    "it": Region("it", (PlateFormat("standard", "LLDDDLL", description="e.g. AB123CD"),), description="Italy"),
    "es": Region("es", (PlateFormat("standard", "DDDDLLL", description="e.g. 1234BCD"),), description="Spain"),
    "nl": Region("nl", (PlateFormat("sidecode", regex=r"^[A-Z0-9]{6}$", description="6 characters in sidecode groups"),), description="Netherlands"),
    "eu": Region("eu", (PlateFormat("alnum", regex=r"^[A-Z0-9]{5,8}$", description="Generic EU 5–8 alphanumerics"),), description="Generic European"),
    "us": Region("us", (PlateFormat("alnum", regex=r"^[A-Z0-9]{2,8}$", description="US states: 2–8 alphanumerics"),), min_length=2, description="United States (generic)"),
    "br": Region("br", (PlateFormat("mercosur", "LLLDLDD", description="e.g. ABC1D23"), PlateFormat("legacy", "LLLDDDD")), description="Brazil"),
    "ae": Region("ae", (PlateFormat("standard", regex=r"^[A-Z]{0,2}[0-9]{1,5}$"),), min_length=1, description="UAE (code + number)"),
}


def load_region(spec: str) -> Region:
    """Region by name, or from a JSON/YAML file path."""
    key = spec.lower().strip()
    if key in REGIONS:
        return REGIONS[key]
    path = Path(spec)
    if not path.is_file():
        from textlens.errors import ConfigurationError

        raise ConfigurationError(f"Unknown plate region {spec!r}.", hint=f"Use one of: {', '.join(sorted(REGIONS))}, or a JSON/YAML file.")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return region_from_dict(data)


def region_from_dict(data: Dict) -> Region:
    fmts = tuple(PlateFormat(f.get("name", f"rule{i}"), f.get("template"), f.get("regex"), f.get("description", ""))
                 for i, f in enumerate(data.get("formats", [])))
    return Region(data.get("name", "custom"), fmts, int(data.get("min_length", 4)), int(data.get("max_length", 10)), data.get("description", ""))
