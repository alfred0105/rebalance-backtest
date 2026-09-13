from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _run_git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=check,
    )


def _repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(proc.stdout.strip())


def publish_latest_report(payload: dict[str, Any], markdown: str) -> tuple[bool, str]:
    """Write the latest report into runs/ and push only those files.

    Backtest results are never mixed with unrelated working-tree changes because
    git add is restricted to runs/latest.json and runs/latest.md.
    """
    try:
        repo_root = _repo_root()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "skipped (not running inside a Git repository)"

    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / "latest.json"
    md_path = runs_dir / "latest.md"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown, encoding="utf-8")

    try:
        _run_git(repo_root, "add", "--", "runs/latest.json", "runs/latest.md")
        staged = _run_git(repo_root, "diff", "--cached", "--quiet", check=False)
        if staged.returncode == 0:
            return True, "already up to date"

        _run_git(
            repo_root,
            "commit",
            "-m",
            "Update latest backtest results [skip ci]",
        )
        push = _run_git(repo_root, "push", "origin", "HEAD", check=False)
        if push.returncode != 0:
            detail = (push.stderr or push.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            return False, f"saved and committed locally, but push failed{suffix}"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            output = (exc.stderr or exc.stdout or "").strip().splitlines()
            if output:
                detail = f": {output[-1]}"
        return False, f"saved locally, but Git publish failed{detail}"

    return True, "published to runs/latest.json and runs/latest.md"
