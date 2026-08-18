"""Per-task git worktrees for parallel execution.

Each task attempt cycle works in an isolated worktree on its own branch:

    workspace/
    ├── repository/            # main checkout = integration branch
    └── worktrees/
        ├── T001-A1/           # branch harness/task/T001/attempt-1
        └── T002-A1/           # branch harness/task/T002/attempt-1

All roles of ONE task attempt (Analyst -> Developer -> Tester ->
Verification -> Reviewer) see the SAME worktree — conversations are
fresh, filesystem state is shared. Different tasks never share a
worktree, so parallel mutation cannot collide.

The harness owns the whole lifecycle; agents are blocked from `git
worktree` (and every other git lifecycle command) by the security hooks.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .repository import GitError, GitRepository

logger = logging.getLogger(__name__)


@dataclass
class WorktreeHandle:
    task_key: str
    cycle: int                 # attempt-cycle number (first attempt_no of the cycle)
    path: Path
    branch: str
    base_commit: str
    repo: GitRepository        # GitRepository rooted at the worktree


class WorktreeManager:
    def __init__(self, main_repo: GitRepository, worktrees_root: Path, branch_prefix: str):
        self.main = main_repo
        self.root = Path(worktrees_root)
        self.branch_prefix = branch_prefix.rstrip("/")

    def branch_name(self, task_key: str, cycle: int) -> str:
        return f"{self.branch_prefix}/{task_key}/attempt-{cycle}"

    def worktree_path(self, task_key: str, cycle: int) -> Path:
        return self.root / f"{task_key}-A{cycle}"

    def create(self, task_key: str, cycle: int, base_commit: str) -> WorktreeHandle:
        """Create the isolated worktree + branch for one task attempt cycle."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.worktree_path(task_key, cycle)
        branch = self.branch_name(task_key, cycle)
        if path.exists():
            # A leftover from a crash the recovery pass already archived —
            # never silently reuse unknown state.
            raise GitError(f"worktree path {path} already exists; recover it first")
        if self.main.branch_exists(branch):
            self.main.delete_branch(branch)
        self.main.add_worktree(path, branch, base_commit)
        return WorktreeHandle(
            task_key=task_key,
            cycle=cycle,
            path=path,
            branch=branch,
            base_commit=base_commit,
            repo=GitRepository(path),
        )

    def remove(self, handle: WorktreeHandle, delete_branch: bool = True) -> None:
        self.remove_path(handle.path, branch=handle.branch if delete_branch else None)

    def remove_path(self, path: str | Path, branch: str | None = None) -> None:
        """Remove one worktree (and optionally its branch). Never touches
        any other worktree."""
        path = Path(path)
        try:
            if path.exists():
                self.main.remove_worktree(path)
        except GitError as exc:
            logger.warning("git worktree remove failed for %s (%s); cleaning up", path, exc)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            self.main.prune_worktrees()
        if branch:
            self.main.delete_branch(branch)

    def registered_paths(self) -> list[Path]:
        return [Path(p) for p in self.main.list_worktrees()]

    def owns(self, path: str | Path, branch: str | None = None) -> bool:
        """True only for worktrees this manager created.

        A repository may legitimately carry worktrees the harness knows
        nothing about (the operator's own checkouts). Recovery must never
        force-remove those or `branch -D` their branch — which, for a clean
        branch holding unmerged commits, would drop its only reference. A
        worktree counts as harness-owned only when it lives under the
        configured worktrees root AND (when a branch is given) sits on a
        branch under the harness branch prefix.
        """
        try:
            resolved = Path(path).resolve()
        except OSError:
            return False
        if not resolved.is_relative_to(self.root.resolve()):
            return False
        if branch is not None and not self.owns_branch(branch):
            return False
        return True

    def owns_branch(self, branch: str | None) -> bool:
        return bool(branch) and branch.startswith(f"{self.branch_prefix}/")

    def managed_paths(self) -> list[Path]:
        """Registered worktrees located under this manager's root."""
        return [path for path in self.registered_paths() if self.owns(path)]
