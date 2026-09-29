"""
textlens.core.extract
─────────────────────
Schema-driven structured extraction on top of any OCR result.

Schemas can be a simple mapping (``{"total": "float", "date": "date"}``),
a JSON Schema object, or a Pydantic model class.

Strategies
    ``vlm``        prompt a document VLM (local or remote) with the schema and
                   parse its JSON answer — best for messy layouts.
    ``heuristic``  fully local: key/value pairs (``Label: value``), labels
                   followed by values on the next line, table rows and typed
                   patterns (dates, amounts), matched to field names with
                   synonyms and fuzzy matching.  Each value keeps page, bbox
                   and confidence for review.
    ``auto``       ``vlm`` when a generative model can run, else ``heuristic``.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from textlens.core.result import Result

_SYNONYMS: Dict[str, List[str]] = {
    "invoice_number": ["invoice #", "invoice no", "invoice number", "inv no", "inv #", "invoice id", "bill no", "bill number"],
    "total": ["total", "total due", "amount due", "grand total", "balance due", "total amount", "amount payable", "net payable"],
    "subtotal": ["subtotal", "sub total", "sub-total", "subotal"],
    "tax": ["tax", "vat", "gst", "sales tax"],
    "date": ["date", "invoice date", "issue date", "issued", "dated", "bill date"],
    "due_date": ["due date", "payment due", "due by"],
    "vendor": ["vendor", "seller", "from", "supplier", "company", "billed by"],
    "customer": ["bill to", "billed to", "customer", "client", "sold to", "ship to"],
    "email": ["email", "e-mail"],
    "phone": ["phone", "tel", "telephone", "mobile"],
}
_TYPE_ALIASES = {
    "str": "string", "string": "string", "text": "string",
    "float": "number", "number": "number", "decimal": "number", "currency": "number", "amount": "number", "money": "number",
    "int": "integer", "integer": "integer",
    "date": "date", "datetime": "date",
    "bool": "boolean", "boolean": "boolean",
}
_DATE_PATTERNS = [
    (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", "ymd"),
    (r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", "dmy_or_mdy"),
    (r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+(\d{4})\b", "d_mon_y"),
    (r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", "mon_d_y"),
]
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTHS["sept"] = 9
_NUMBER_RE = re.compile(r"[-+]?\(?[$€£₹¥]?\s*\d[\d,.\s]*\d?\)?")


@dataclass
class FieldSpec:
    name: str
    type: str = "string"
    description: str = ""


@dataclass
class Extraction:
    data: Dict[str, Any]
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    method: str = "heuristic"
    result: Optional[Result] = None
    warnings: List[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {"data": self.data, "fields": self.fields, "method": self.method, "warnings": self.warnings}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


def normalize_schema(schema: Any) -> List[FieldSpec]:
    if schema is None:
        return []
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    if isinstance(schema, str):
        schema = json.loads(schema)
    if isinstance(schema, (list, tuple)):
        return [FieldSpec(str(n), infer_type(str(n))) for n in schema]
    if isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema:
        out = []
        for name, prop in schema["properties"].items():
            t = prop.get("type", "string") if isinstance(prop, dict) else "string"
            if isinstance(prop, dict) and prop.get("format") in ("date", "date-time"):
                t = "date"
            out.append(FieldSpec(name, _TYPE_ALIASES.get(str(t), "string"), prop.get("description", "") if isinstance(prop, dict) else ""))
        return out
    if isinstance(schema, dict):
        return [FieldSpec(str(k), _TYPE_ALIASES.get(str(v).lower(), "string")) for k, v in schema.items()]
    raise ValueError("schema must be a mapping, JSON Schema, list of names or Pydantic model")


# ── typing helpers ───────────────────────────────────────────────────────────

_NUMBER_NAMES = ("total", "amount", "subtotal", "sub_total", "tax", "vat", "gst", "price", "cost", "balance", "fee", "discount", "sum")
_INT_NAMES = ("quantity", "qty", "count", "pages")
_DATE_NAMES = ("date", "dob", "birthday", "expiry", "expires", "issued", "due")


def infer_type(name: str) -> str:
    """Guess a field type from its name when the schema gives none."""
    n = name.lower()
    tokens = set(re.split(r"[^a-z]+", n))
    if tokens & set(_DATE_NAMES) or n.endswith("_date"):
        return "date"
    if tokens & set(_NUMBER_NAMES):
        return "number"
    if tokens & set(_INT_NAMES):
        return "integer"
    return "string"



def parse_date(text: str) -> Optional[str]:
    t = text.lower()
    for pattern, kind in _DATE_PATTERNS:
        m = re.search(pattern, t)
        if not m:
            continue
        try:
            if kind == "ymd":
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif kind == "dmy_or_mdy":
                a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                d, mo = (a, b) if a > 12 else (b, a) if b > 12 else (a, b)  # ambiguous → day-first
            elif kind == "d_mon_y":
                d, mo, y = int(m.group(1)), _MONTHS[m.group(2)[:4] if m.group(2).startswith("sept") else m.group(2)[:3]], int(m.group(3))
            else:
                mo, d, y = _MONTHS[m.group(1)[:4] if m.group(1).startswith("sept") else m.group(1)[:3]], int(m.group(2)), int(m.group(3))
            return date(y, mo, d).isoformat()
        except (ValueError, KeyError):
            continue
    return None


def parse_number(text: str) -> Optional[float]:
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    raw = m.group(0)
    negative = raw.strip().startswith(("-", "(")) and (raw.strip().endswith(")") or raw.strip().startswith("-"))
    digits = re.sub(r"[^\d.,]", "", raw)
    if not digits:
        return None
    if "," in digits and "." in digits:
        if digits.rfind(",") > digits.rfind("."):  # 1.234,56
            digits = digits.replace(".", "").replace(",", ".")
        else:  # 1,234.56
            digits = digits.replace(",", "")
    elif "," in digits:
        parts = digits.split(",")
        digits = digits.replace(",", ".") if len(parts[-1]) in (1, 2) and len(parts) == 2 else digits.replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def coerce(value: str, type_: str) -> Any:
    value = value.strip()
    if type_ == "number":
        return parse_number(value)
    if type_ == "integer":
        n = parse_number(value)
        return int(n) if n is not None else None
    if type_ == "date":
        return parse_date(value)
    if type_ == "boolean":
        return value.lower() in ("yes", "true", "y", "1", "x", "✓", "checked")
    return value or None


# ── heuristic extraction ─────────────────────────────────────────────────────


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9#]+", " ", s.lower()).strip()


def _label_score(field_name: str, label: str) -> float:
    label_n = _norm(label)
    if not label_n:
        return 0.0
    candidates = [field_name.replace("_", " ")] + _SYNONYMS.get(field_name.lower(), [])
    best = 0.0
    for cand in candidates:
        c = _norm(cand)
        if label_n == c:
            return 1.0
        if c and (label_n.endswith(c) or label_n.startswith(c)):
            best = max(best, 0.85)
        best = max(best, difflib.SequenceMatcher(None, label_n, c).ratio() * 0.9)
    return best


def _pairs(result: Result) -> List[Tuple[str, str, int, Any, Optional[float]]]:
    """Candidate (label, value, page, bbox, confidence) pairs from a result."""
    pairs = []
    for page in result.pages:
        lines = page.lines
        for i, ln in enumerate(lines):
            text = ln.text.strip()
            m = re.match(r"^\s*([^:：]{1,40})[:：]\s*(.+)$", text)
            if m:
                pairs.append((m.group(1), m.group(2), page.number, ln.bbox, ln.confidence))
            elif "\t" in text:
                label, _, value = text.partition("\t")
                pairs.append((label, value.strip(), page.number, ln.bbox, ln.confidence))
            else:
                # "TOTAL DUE $5.00": label words followed by an amount/date token.
                m2 = re.match(r"^([A-Za-z][A-Za-z #/.-]{1,40}?)\s+([-$€£₹¥(]?\d[\d,./-]*\)?%?)$", text)
                if m2:
                    pairs.append((m2.group(1), m2.group(2), page.number, ln.bbox, ln.confidence))
                elif text.endswith(":") and i + 1 < len(lines):
                    nxt = lines[i + 1]
                    pairs.append((text[:-1], nxt.text.strip(), page.number, nxt.bbox, nxt.confidence))
        for table in page.tables:
            grid = table.grid()
            for row in grid:
                if len(row) >= 2 and row[0].strip():
                    pairs.append((row[0], row[-1], page.number, table.bbox, table.confidence))
            if len(grid) >= 2:
                for c, header in enumerate(grid[0]):
                    if header.strip() and c < len(grid[1]):
                        pairs.append((header, grid[1][c], page.number, table.bbox, table.confidence))
    return pairs


def extract_heuristic(result: Result, fields: List[FieldSpec]) -> Extraction:
    pairs = _pairs(result)
    data: Dict[str, Any] = {}
    details: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        best = None
        for label, value, page, bbox, conf in pairs:
            score = _label_score(f.name, label)
            if score < 0.6:
                continue
            typed = coerce(value, f.type)
            if typed is None:
                continue
            key = (score, conf or 0.0)
            if best is None or key > best[0]:
                best = (key, typed, value, page, bbox, conf, label)
        if best is None and f.type == "date":
            for page in result.pages:
                for ln in page.lines:
                    d = parse_date(ln.text)
                    if d:
                        best = ((0.5, ln.confidence or 0.0), d, ln.text, page.number, ln.bbox, ln.confidence, "date pattern")
                        break
                if best:
                    break
        if best is None and f.name.lower() in ("vendor", "company", "title", "seller"):
            first = next((b for b in result.blocks if b.type in ("title", "heading") and b.text.strip()), None)
            if first is not None:
                best = ((0.4, first.confidence or 0.0), first.text.strip(), first.text, first.page, first.bbox, first.confidence, "document title")
        if best is None:
            data[f.name] = None
            details[f.name] = {"value": None, "method": "heuristic", "found": False}
            continue
        (score, _), typed, raw, page, bbox, conf, label = best
        data[f.name] = typed
        details[f.name] = {
            "value": typed,
            "raw": raw,
            "label": label,
            "page": page,
            "bbox": list(bbox) if bbox else None,
            "confidence": round(score * (conf if conf is not None else 1.0), 3),
            "method": "heuristic",
            "found": True,
        }
    return Extraction(data=data, fields=details, method="heuristic", result=result)


# ── VLM extraction ───────────────────────────────────────────────────────────


def build_prompt(fields: List[FieldSpec]) -> str:
    spec = {f.name: (f.type + (f" — {f.description}" if f.description else "")) for f in fields}
    return (
        "Extract the following fields from this document and answer with a single JSON object only, "
        "no prose and no code fences. Use null for fields that are not present. Dates as YYYY-MM-DD, "
        f"numbers without currency symbols.\nFields: {json.dumps(spec, ensure_ascii=False)}"
    )


def parse_json_answer(text: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
