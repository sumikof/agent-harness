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

    def _run(
        self, *args: str, check: bool = True, input_text: str | None = None
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=str(self.path),
            capture_output=True,
            text=True,
            input=input_text,
        )
        if check and result.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def is_repo(self) -> bool:
        """True only when path is the ROOT of a git work tree.

        A nested directory inside another checkout also reports
        --is-inside-work-tree=true, but add/reset/commit from there would
        mutate the enclosing repository — never accept that.
        """
        if not self.path.exists():
            return False
        result = self._run("rev-parse", "--show-toplevel", check=False)
        if result.returncode != 0:
            return False
        toplevel = result.stdout.strip()
        return bool(toplevel) and Path(toplevel).resolve() == self.path.resolve()

    def init(self, initial_branch: str = "main") -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._run("init", "-b", initial_branch)

    def head_commit(self) -> str | None:
        result = self._run("rev-parse", "HEAD", check=False)
        if result.returncode != 0:
            return None  # no commits yet
        return result.stdout.strip()

    def current_branch(self) -> str:
        # symbolic-ref works on unborn branches too; a detached HEAD has no
        # symbolic ref and is reported as such.
        result = self._run("symbolic-ref", "--short", "HEAD", check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        return "DETACHED"

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

    def untracked_paths(self) -> list[str]:
        """Untracked paths, parsed from NUL-separated porcelain output so
        names with spaces, quotes, or non-ASCII characters survive intact."""
        out = self._run("status", "--porcelain", "-z").stdout
        fields = out.split("\0")
        untracked: list[str] = []
        index = 0
        while index < len(fields):
            entry = fields[index]
            if not entry:
                index += 1
                continue
            status = entry[:2]
            if status == "??":
                untracked.append(entry[3:])
            # Rename/copy entries carry the source path as an extra field.
            if "R" in status or "C" in status:
                index += 2
            else:
                index += 1
        return untracked

    def dirty_diff_readonly(self) -> str:
        """full_dirty_diff() that leaves the index as it found it.

        Read-only inspection (e.g. recovery deciding whether a tree may be
        reset) must not convert the user's untracked files into
        intent-to-add entries; the ita entries created for the diff are
        removed again afterwards.
        """
        newly_untracked = self.untracked_paths()
        diff = self.full_dirty_diff()
        if newly_untracked:
            if self.head_commit():
                self._run("reset", "-q", "--", *newly_untracked)
            else:
                # Unborn HEAD: drop the ita index entries directly. Porcelain
                # may report whole directories ('?? dir/'), so removal must
                # be recursive.
                self._run("rm", "--cached", "-r", "-q", "--", *newly_untracked)
        return diff

    def snapshot_dirty(self) -> str:
        """Binary-safe patch of everything uncommitted (incl. untracked).

        Round-trippable via apply_patch(): reset_hard() + apply_patch()
        restores the worktree to exactly this state.
        """
        self._run("add", "-A", "-N", check=False)  # intent-to-add so untracked shows
        if self.head_commit():
            return self._run("diff", "--binary", "--no-color", "HEAD").stdout
        return self._run("diff", "--binary", "--no-color").stdout

    def apply_patch(self, patch: str) -> None:
        self._run("apply", "--whitespace=nowarn", input_text=patch)

    def changed_files(self) -> list[str]:
        out = self._run("status", "--porcelain").stdout
        files = []
        for line in out.splitlines():
            if len(line) > 3:
                files.append(line[3:].split(" -> ")[-1].strip())
        return files

    def add_all(self) -> None:
        self._run("add", "-A")

    def commit(
        self, message: str, allow_empty: bool = False, trailers: dict[str, str] | None = None
    ) -> str:
        if trailers:
            trailer_block = "\n".join(f"{key}: {value}" for key, value in trailers.items())
            message = f"{message}\n\n{trailer_block}"
        args = [
            "-c", "user.name=agent-harness",
            "-c", "user.email=agent-harness@localhost",
            "commit", "-m", message,
        ]
        if allow_empty:
            args.append("--allow-empty")
        self._run(*args)
        return self.head_commit() or ""

    def find_commit_by_trailer(self, key: str, value: str, limit: int = 500) -> str | None:
        """Find a recent commit whose message carries `key: value`.

        Used by crash recovery to decide whether a journaled GIT_COMMIT
        intent was already executed before the process died.
        """
        result = self._run(
            "log", f"-{limit}", "--fixed-strings", f"--grep={key}: {value}",
            "--format=%H", check=False,
        )
        if result.returncode != 0:
            return None
        commits = result.stdout.split()
        return commits[0] if commits else None

    def commit_message(self, ref: str) -> str:
        result = self._run("log", "-1", "--format=%B", ref, check=False)
        return result.stdout if result.returncode == 0 else ""

    def commit_exists(self, ref: str) -> bool:
        result = self._run("cat-file", "-e", f"{ref}^{{commit}}", check=False)
        return result.returncode == 0

    def reset_hard(self, ref: str = "HEAD") -> None:
        # `clean -fd` deliberately leaves ignored files alone: build caches
        # and pre-existing user data (via .gitignore / info/exclude) are not
        # the harness's to destroy.
        self._run("reset", "--hard", ref)
        self._run("clean", "-fd")

    def log_oneline(self, limit: int = 20) -> str:
        return self._run("log", "--oneline", f"-{limit}", check=False).stdout
