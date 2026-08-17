"""Git commit as the official task checkpoint.

The harness — never an agent — commits, exactly once per PASSed task.
SQLite records the task -> commit mapping so the last good state is
always recoverable.

When an OperationRepository is wired in, every checkpoint commit is
journaled as GIT_COMMIT_INTENT -> git commit -> GIT_COMMIT_RESULT, and
the commit itself carries Harness-Task / Harness-Operation-Id trailers.
A crash between the git commit and the DB update is then recoverable:
recovery finds the pending intent, locates the commit by its trailer,
and reconciles the DB instead of committing twice.
"""

from __future__ import annotations

import hashlib

from ..database.operation_repository import OperationRepository, OperationType
from .repository import GitRepository

TASK_TRAILER = "Harness-Task"
OPERATION_TRAILER = "Harness-Operation-Id"


class CheckpointManager:
    def __init__(self, repo: GitRepository, operations: OperationRepository | None = None):
        self.repo = repo
        self.operations = operations
        # operation_id of the last journaled commit. The RESULT for it is
        # deliberately NOT recorded here: the caller records it in the same
        # transaction as the task-completion updates, so there is no window
        # where the operation looks settled but the task state was lost.
        self.pending_operation_id: str | None = None

    def commit_task(
        self,
        task_key: str,
        title: str,
        *,
        project_id: int | None = None,
        task_id: int | None = None,
        attempt_id: int | None = None,
    ) -> str | None:
        """Commit all working-tree changes for a passed task.

        Returns the new commit hash, or None if there was nothing to commit
        (a task may legitimately produce no diff, e.g. verification-only).

        When journaled, the operation is left PENDING with its id in
        `pending_operation_id`; the caller must record the result together
        with the task-state updates (one transaction). A crash before that
        leaves the intent PENDING and the commit discoverable by trailer,
        which is exactly what recovery reconciles.
        """
        self.pending_operation_id = None
        if not self.repo.is_dirty():
            return None
        message = f"agent({task_key}): {title}"

        if self.operations is None:
            self.repo.add_all()
            return self.repo.commit(message)

        base_head = self.repo.head_commit()
        diff = self.repo.full_dirty_diff()
        diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        operation_id = self.operations.record_intent(
            OperationType.GIT_COMMIT,
            {
                "task_key": task_key,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "base_head": base_head,
                "diff_sha256": diff_hash,
                "commit_message": message,
            },
            project_id=project_id,
            task_id=task_id,
            attempt_id=attempt_id,
        )
        # Intent is durably committed before the side effect below.
        self.repo.add_all()
        commit_hash = self.repo.commit(
            message,
            trailers={TASK_TRAILER: task_key, OPERATION_TRAILER: operation_id},
        )
        self.pending_operation_id = operation_id
        return commit_hash

    def find_committed_operation(self, operation_id: str) -> str | None:
        """The commit hash a journaled GIT_COMMIT intent produced, if any."""
        return self.repo.find_commit_by_trailer(OPERATION_TRAILER, operation_id)

    def rollback_to(self, commit_hash: str) -> None:
        self.repo.reset_hard(commit_hash)

    def discard_working_tree(self) -> None:
        # ensure_project() guarantees a baseline commit before any agent
        # runs, so HEAD normally exists. With no commits there is nothing
        # safe to reset to — pre-existing (possibly ignored) user data must
        # never be destroyed — so this is a no-op then.
        if self.repo.head_commit() is not None:
            self.repo.reset_hard("HEAD")
