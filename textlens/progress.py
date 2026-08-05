"""
textlens.progress
────────────────
Terminal progress bars, execution timers, and real-time status output utilities for TextLens.
"""

from __future__ import annotations

import sys
import time
from typing import Optional


class ProgressTracker:
    """Real-time progress tracker and timer for OCR processing."""

    def __init__(self, total_steps: int = 100, desc: str = "Processing") -> None:
        self.total_steps = max(1, total_steps)
        self.desc = desc
        self.start_time = time.time()
        self.current_step = 0

    def update(self, step: int, message: Optional[str] = None) -> None:
        """Update progress step and display progress bar in terminal."""
        self.current_step = min(step, self.total_steps)
        percent = int((self.current_step / self.total_steps) * 100)
        bar_len = 25
        filled_len = int(bar_len * self.current_step // self.total_steps)
        bar = "█" * filled_len + "░" * (bar_len - filled_len)

        elapsed = time.time() - self.start_time
        msg_str = f" - {message}" if message else ""

        sys.stdout.write(f"\r[TextLens] {self.desc} [{bar}] {percent}% ({elapsed:.2f}s){msg_str}")
        sys.stdout.flush()

        if self.current_step >= self.total_steps:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def complete(self, message: str = "Done!") -> float:
        """Mark processing complete and return total elapsed seconds."""
        elapsed = round(time.time() - self.start_time, 3)
        self.update(self.total_steps, message=f"{message} (Total: {elapsed}s)")
        return elapsed


def print_step(step_num: int, total: int, title: str) -> None:
    """Print clean step header."""
    print(f"\n⚡ [TextLens {step_num}/{total}] {title}")
