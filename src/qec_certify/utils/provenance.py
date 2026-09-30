"""Provenance metadata saved alongside results."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit_hash(repo_dir: str | Path | None = None) -> str | None:
    """Current ``HEAD`` commit, suffixed with ``-dirty`` if the tree has changes.

    Returns ``None`` outside a git checkout.
    """
    cwd = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit
