"""Thin subprocess wrapper around git for the target repository.

Only the harness runs these operations; agents are blocked from git
lifecycle commands by the security hooks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(Exception):
    pass


class GitRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=str(self.path),
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def is_repo(self) -> bool:
        if not self.path.exists():
            return False
        result = self._run("rev-parse", "--is-inside-work-tree", check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def init(self, initial_branch: str = "main") -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._run("init", "-b", initial_branch)

    def head_commit(self) -> str | None:
        result = self._run("rev-parse", "HEAD", check=False)
        if result.returncode != 0:
            return None  # no commits yet
        return result.stdout.strip()

    def current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def is_dirty(self) -> bool:
        result = self._run("status", "--porcelain")
        return bool(result.stdout.strip())

    def status(self) -> str:
        return self._run("status", "--porcelain").stdout

    def diff(self, staged: bool = False) -> str:
        args = ["diff", "--no-color"]
        if staged:
            args.append("--cached")
        return self._run(*args).stdout

    def full_dirty_diff(self) -> str:
        """Diff of everything uncommitted, including untracked files."""
        self._run("add", "-A", "-N", check=False)  # intent-to-add so untracked shows in diff
        if self.head_commit():
            return self._run("diff", "--no-color", "HEAD").stdout
        return self._run("diff", "--no-color").stdout

    def changed_files(self) -> list[str]:
        out = self._run("status", "--porcelain").stdout
        files = []
        for line in out.splitlines():
            if len(line) > 3:
                files.append(line[3:].split(" -> ")[-1].strip())
        return files

    def add_all(self) -> None:
        self._run("add", "-A")

    def commit(self, message: str) -> str:
        self._run(
            "-c", "user.name=agent-harness",
            "-c", "user.email=agent-harness@localhost",
            "commit", "-m", message,
        )
        return self.head_commit() or ""

    def reset_hard(self, ref: str = "HEAD") -> None:
        self._run("reset", "--hard", ref)
        self._run("clean", "-fd")

    def log_oneline(self, limit: int = 20) -> str:
        return self._run("log", "--oneline", f"-{limit}", check=False).stdout
