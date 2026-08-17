"""Git commit as the official task checkpoint.

The harness — never an agent — commits, exactly once per PASSed task.
SQLite records the task -> commit mapping so the last good state is
always recoverable.
"""

from __future__ import annotations

from .repository import GitRepository


class CheckpointManager:
    def __init__(self, repo: GitRepository):
        self.repo = repo

    def commit_task(self, task_key: str, title: str) -> str | None:
        """Commit all working-tree changes for a passed task.

        Returns the new commit hash, or None if there was nothing to commit
        (a task may legitimately produce no diff, e.g. verification-only).
        """
        if not self.repo.is_dirty():
            return None
        self.repo.add_all()
        message = f"agent({task_key}): {title}"
        return self.repo.commit(message)

    def rollback_to(self, commit_hash: str) -> None:
        self.repo.reset_hard(commit_hash)

    def discard_working_tree(self) -> None:
        # ensure_project() guarantees a baseline commit before any agent
        # runs, so HEAD normally exists. With no commits there is nothing
        # safe to reset to — pre-existing (possibly ignored) user data must
        # never be destroyed — so this is a no-op then.
        if self.repo.head_commit() is not None:
            self.repo.reset_hard("HEAD")
