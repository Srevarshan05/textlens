"""Console helpers shared by CLI commands (Rich is optional)."""

from __future__ import annotations

import io
import json
import logging
import sys
from typing import Any, List, Optional


def setup_streams() -> None:
    """Make stdout/stderr safe for any Unicode on every platform.

    Pipes and files get UTF-8; interactive consoles keep their encoding but
    replace unencodable characters instead of crashing (Windows cp1252).
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if not isinstance(stream, io.TextIOWrapper):
            continue
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def setup_logging(verbosity: int, quiet: bool = False) -> None:
    from textlens.config import get_settings

    level_name = get_settings().log_level
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = getattr(logging, level_name, logging.WARNING)
    from textlens.observability import configure_logging

    configure_logging(level=level)


def console(stderr: bool = False) -> Optional[Any]:
    try:
        from rich.console import Console

        return Console(stderr=stderr, highlight=False, soft_wrap=False)
    except ImportError:
        return None


def info(msg: str) -> None:
    c = console(stderr=True)
    if c:
        c.print(msg)
    else:
        print(_strip_markup(msg), file=sys.stderr)


def error(exc: BaseException) -> None:
    message = getattr(exc, "message", None) or str(exc)
    hint = getattr(exc, "hint", None)
    c = console(stderr=True)
    if c:
        from rich.markup import escape

        c.print(f"[bold red]Error:[/bold red] {escape(message.strip())}")
        if hint:
            c.print(f"[yellow]Hint:[/yellow] {escape(hint)}")
    else:
        print(f"Error: {message.strip()}", file=sys.stderr)
        if hint:
            print(f"Hint: {hint}", file=sys.stderr)


def _strip_markup(msg: str) -> str:
    import re

    return re.sub(r"\[/?[a-z ]+[^\]]*\]", "", msg)


def emit(text: str, output: Optional[str] = None) -> None:
    """Write command output to a file or stdout."""
    if output and output != "-":
        from pathlib import Path

        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        info(f"[green]✓[/green] wrote {path}")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()


def emit_json(data: Any, output: Optional[str] = None) -> None:
    emit(json.dumps(data, indent=2, ensure_ascii=False, default=str), output)


def parse_pages(spec: Optional[str]) -> Optional[List[int]]:
    """Parse ``"1-3,5,8-"`` style page selections (1-indexed)."""
    if not spec:
        return None
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start = int(a) if a else 1
            end = int(b) if b else start + 10_000
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(p for p in pages if p >= 1))
