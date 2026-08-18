"""Thin subprocess wrapper around git for the target repository.

Only the harness runs these operations; agents are blocked from git
lifecycle commands by the security hooks.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


class GitError(Exception):
    pass


@dataclass
class WorktreeSnapshot:
    """Byte-exact capture of every uncommitted change: the actual on-disk
    files (tarred, untouched by git clean filters) plus the deletions.
    Restorable via GitRepository.restore_worktree_state()."""

    tar_bytes: bytes
    deleted: list[str] = field(default_factory=list)
    state_hash: str = ""


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
        return self.dirty_diff_readonly_bytes().decode("utf-8", errors="replace")

    def dirty_diff_readonly_bytes(self) -> bytes:
        """snapshot_dirty_bytes() that leaves the index as it found it.

        Read-only inspection (e.g. recovery deciding whether a tree may be
        reset) must not convert the user's untracked files into
        intent-to-add entries; the ita entries created for the diff are
        removed again afterwards. Raw bytes, filter-free — the SAME basis
        as the archived snapshot, so evidence hashes always compare like
        with like regardless of file encodings.
        """
        newly_untracked = self.untracked_paths()
        diff = self.snapshot_dirty_bytes()
        if newly_untracked:
            if self.head_commit():
                self._run("reset", "-q", "--", *newly_untracked)
            else:
                # Unborn HEAD: drop the ita index entries directly. Porcelain
                # may report whole directories ('?? dir/'), so removal must
                # be recursive.
                self._run("rm", "--cached", "-r", "-q", "--", *newly_untracked)
        return diff

    def changed_paths(self) -> list[str]:
        """Every path with uncommitted changes (untracked included), parsed
        NUL-safely. Directories may appear as 'dir/' entries; rename/copy
        entries contribute both sides."""
        out = self._run("status", "--porcelain", "-z").stdout
        fields = out.split("\0")
        paths: list[str] = []
        index = 0
        while index < len(fields):
            entry = fields[index]
            if not entry:
                index += 1
                continue
            status = entry[:2]
            paths.append(entry[3:])
            if "R" in status or "C" in status:
                index += 1
                if index < len(fields) and fields[index]:
                    paths.append(fields[index])  # rename/copy source
            index += 1
        return paths

    def expanded_changed_files(self) -> list[str]:
        """changed_paths() with untracked directories expanded to files.

        Symlinks to directories inside an untracked tree appear in
        os.walk()'s dirs list (never descended, followlinks=False) — they
        are entries in their own right and must not be dropped.
        """
        files: set[str] = set()
        for rel in self.changed_paths():
            full = self.path / rel
            if full.is_dir() and not full.is_symlink():
                for root, dirs, names in os.walk(full):
                    for name in names:
                        files.add(str((Path(root) / name).relative_to(self.path)))
                    for name in dirs:
                        candidate = Path(root) / name
                        if candidate.is_symlink():
                            files.add(str(candidate.relative_to(self.path)))
            else:
                files.add(rel.rstrip("/"))
        return sorted(files)

    def dirty_state_hash(self) -> str:
        """Fingerprint of the ACTUAL uncommitted worktree state.

        Computed from the real on-disk bytes of every changed path (plus
        deletion markers), never from git diff output — clean filters,
        textconv, and encodings cannot distort it. This is the single basis
        for all recovery evidence hashes.
        """
        digest = hashlib.sha256()
        for rel in self.expanded_changed_files():
            full = self.path / rel
            digest.update(rel.encode("utf-8", "surrogateescape"))
            digest.update(b"\0")
            if full.is_symlink():
                digest.update(b"L")
                digest.update(os.readlink(full).encode("utf-8", "surrogateescape"))
            elif full.is_file():
                # Git tracks the executable bit — a mode-only change is a
                # real state difference and must not hash alike.
                executable = bool(full.stat().st_mode & 0o100)
                digest.update(b"X" if executable else b"F")
                digest.update(full.read_bytes())
            else:
                try:
                    node = os.lstat(full)
                except (FileNotFoundError, NotADirectoryError):
                    digest.update(b"D")  # genuinely deleted / missing
                else:
                    # A special node (FIFO, socket, device) at the path is a
                    # different state than a deletion — never hash alike, or
                    # stale evidence could get a user's node reset away. The
                    # fingerprint covers type, permissions, and node identity
                    # (inode/device/mtime), so replacing or chmod-ing the
                    # node changes the hash.
                    digest.update(b"N")
                    for value in (node.st_mode, node.st_ino, node.st_dev,
                                  node.st_mtime_ns):
                        digest.update(int(value).to_bytes(16, "little", signed=False))
            digest.update(b"\0")
        return digest.hexdigest()

    def snapshot_worktree_state(self) -> WorktreeSnapshot:
        """Byte-exact snapshot of all uncommitted changes (files + deletions),
        taken from the filesystem directly — no git filters involved."""
        files = self.expanded_changed_files()
        deleted: list[str] = []
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for rel in files:
                full = self.path / rel
                if full.is_symlink() or full.is_file():
                    tar.add(full, arcname=rel, recursive=False)
                    continue
                try:
                    node = os.lstat(full)
                except (FileNotFoundError, NotADirectoryError):
                    deleted.append(rel)
                    continue
                if stat.S_ISFIFO(node.st_mode):
                    tar.add(full, arcname=rel, recursive=False)  # tar supports FIFOs
                else:
                    # Sockets/devices cannot be captured faithfully — fail
                    # loudly instead of recording them as deletions and
                    # silently dropping them on restore.
                    raise GitError(
                        f"cannot snapshot special node at {rel}; "
                        "unsupported worktree state"
                    )
        return WorktreeSnapshot(
            tar_bytes=buffer.getvalue(), deleted=deleted,
            state_hash=self.dirty_state_hash(),
        )

    def restore_worktree_state(self, snapshot: WorktreeSnapshot) -> None:
        """reset to HEAD, then re-apply a snapshot byte-exactly."""
        self.reset_hard("HEAD")
        if snapshot.tar_bytes:
            with tarfile.open(fileobj=io.BytesIO(snapshot.tar_bytes)) as tar:
                # The snapshot may replace a HEAD file with a directory (or
                # vice versa); clear conflicting HEAD paths before extracting.
                for member in tar.getmembers():
                    self._clear_conflicting_paths(member.name)
                try:
                    # 'tar' (not 'data'): the members are self-authored
                    # relative paths, and 'data' refuses FIFO members.
                    tar.extractall(self.path, filter="tar")
                except TypeError:  # Python without the filter parameter
                    tar.extractall(self.path)
        for rel in snapshot.deleted:
            full = self.path / rel
            if full.is_file() or full.is_symlink():
                full.unlink()

    def _clear_conflicting_paths(self, rel: str) -> None:
        """Remove HEAD paths that block extracting `rel` as a file: an
        ancestor that exists as a file, or the target existing as a dir."""
        parts = PurePosixPath(rel).parts
        for depth in range(1, len(parts)):
            ancestor = self.path.joinpath(*parts[:depth])
            if ancestor.is_symlink() or ancestor.is_file():
                ancestor.unlink()
        target = self.path / rel
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.is_symlink() or target.exists():
            # A plain file blocking a FIFO member (etc.) — clear it so the
            # extraction can recreate the node type from the snapshot.
            target.unlink()

    def snapshot_dirty(self) -> str:
        """Binary-safe patch of everything uncommitted (incl. untracked).

        Round-trippable via apply_patch(): reset_hard() + apply_patch()
        restores the worktree to exactly this state. Repository-configured
        diff transformations (textconv drivers, external diff) are disabled
        — their output is presentation-only and NOT re-applicable, which
        would silently corrupt the archive.
        """
        return self.snapshot_dirty_bytes().decode("utf-8", errors="replace")

    def snapshot_dirty_bytes(self) -> bytes:
        """snapshot_dirty() as raw bytes.

        Diff content is arbitrary bytes (files need not be UTF-8); the
        recovery archive must preserve them exactly, so patches are
        captured, stored, and re-applied without any text decoding.
        """
        self._run("add", "-A", "-N", check=False)  # intent-to-add so untracked shows
        args = ["diff", "--binary", "--no-color", "--no-textconv", "--no-ext-diff"]
        if self.head_commit():
            args.append("HEAD")
        result = subprocess.run(
            ["git", *args], cwd=str(self.path), capture_output=True
        )
        if result.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed: "
                f"{result.stderr.decode('utf-8', errors='replace').strip()}"
            )
        return result.stdout

    def apply_patch(self, patch: str) -> None:
        self.apply_patch_bytes(patch.encode("utf-8"))

    def apply_patch_bytes(self, patch: bytes) -> None:
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            cwd=str(self.path), capture_output=True, input=patch,
        )
        if result.returncode != 0:
            raise GitError(
                "git apply failed: "
                f"{result.stderr.decode('utf-8', errors='replace').strip()}"
            )

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
        # --all: harness commits may live on task branches (worktrees) that
        # are not ancestors of the current checkout.
        result = self._run(
            "log", "--all", f"-{limit}", "--fixed-strings", f"--grep={key}: {value}",
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

    # -- worktrees ---------------------------------------------------------

    def add_worktree(self, path: str | Path, branch: str, base_ref: str) -> None:
        """Create an isolated worktree on a NEW branch at base_ref."""
        self._run("worktree", "add", "-b", branch, str(path), base_ref)

    def add_detached_worktree(self, path: str | Path, commit: str) -> None:
        """Create a worktree pinned to `commit` with NO branch.

        A read-only snapshot: nothing can be committed to it and there is no
        ref to clean up afterwards.
        """
        self._run("worktree", "add", "--detach", str(path), commit)

    def remove_worktree(self, path: str | Path, force: bool = True) -> None:
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path))
        self._run(*args)
        self._run("worktree", "prune", check=False)

    def list_worktrees(self) -> list[str]:
        """Paths of all linked worktrees (the main checkout excluded)."""
        out = self._run("worktree", "list", "--porcelain").stdout
        paths = [
            line[len("worktree "):]
            for line in out.splitlines()
            if line.startswith("worktree ")
        ]
        return [p for p in paths if Path(p).resolve() != self.path.resolve()]

    def prune_worktrees(self) -> None:
        self._run("worktree", "prune", check=False)

    def branch_exists(self, branch: str) -> bool:
        result = self._run(
            "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False
        )
        return result.returncode == 0

    def delete_branch(self, branch: str) -> None:
        self._run("branch", "-D", branch, check=False)

    def rev_parse(self, ref: str) -> str | None:
        result = self._run("rev-parse", ref, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    # -- merges (serialized integration) -----------------------------------

    def merge_no_ff(self, ref: str, message: str, trailers: dict[str, str] | None = None) -> str:
        """Merge `ref` into the current branch with an explicit merge commit.

        Raises MergeConflict (a GitError subclass) on conflicts, leaving the
        merge in progress — the caller decides between resolve and abort.
        """
        if trailers:
            trailer_block = "\n".join(f"{key}: {value}" for key, value in trailers.items())
            message = f"{message}\n\n{trailer_block}"
        result = self._run(
            "-c", "user.name=agent-harness",
            "-c", "user.email=agent-harness@localhost",
            "merge", "--no-ff", "-m", message, ref,
            check=False,
        )
        if result.returncode != 0:
            detail = f"{result.stdout.strip()} {result.stderr.strip()}".strip()
            # Only an actual conflict is a conflict. A stale index.lock, a
            # full disk or an object-store error also exit non-zero but leave
            # no conflicted paths and no merge in progress; classifying those
            # as CONFLICT would archive-and-discard a reviewed change and
            # burn a repair attempt on work that had no real conflict.
            if self.merge_in_progress() or self.conflicted_paths():
                raise MergeConflict(f"merge of {ref} failed: {detail}")
            raise GitError(f"merge of {ref} failed without conflict state: {detail}")
        return self.head_commit() or ""

    def pin_ref(self, name: str, commit: str) -> None:
        """Keep `commit` reachable under refs/... after its branch is gone."""
        self._run("update-ref", name, commit, check=False)

    def merge_in_progress(self) -> bool:
        git_dir = self._run("rev-parse", "--git-dir").stdout.strip()
        return (self.path / git_dir / "MERGE_HEAD").exists()

    def merge_abort(self) -> None:
        self._run("merge", "--abort", check=False)

    def conflicted_paths(self) -> list[str]:
        out = self._run("diff", "--name-only", "--diff-filter=U", check=False).stdout
        return [line for line in out.splitlines() if line.strip()]


class MergeConflict(GitError):
    """A merge could not complete cleanly; the caller owns abort/resolve."""
