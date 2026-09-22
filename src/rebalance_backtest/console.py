from __future__ import annotations

import shutil
import sys


def live_status(message: str) -> None:
    """Rewrite one terminal line in place without stacking progress output."""
    width = max(40, shutil.get_terminal_size((120, 20)).columns)
    text = str(message).replace("\n", " ")
    if len(text) >= width:
        text = text[: width - 4] + "..."
    sys.stdout.write("\r" + text.ljust(width - 1))
    sys.stdout.flush()


def finish_status(message: str) -> None:
    live_status(message)
    sys.stdout.write("\n")
    sys.stdout.flush()
