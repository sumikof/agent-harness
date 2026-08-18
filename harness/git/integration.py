"""Serialized integration of passed task branches.

Task coding runs in parallel; integration into the integration branch
never does. All merges funnel through ONE IntegrationManager holding an
asyncio lock (and the git_integration resource pool is fixed at 1 slot),
so two tasks can never interleave partial merges.

Conflicts are a NORMAL outcome, not an error: independent tasks that
started from the same base commit may collide when the later one merges.
The merge is aborted cleanly, the task transitions to
INTEGRATION_CONFLICT, and a fresh repair attempt re-implements the task
on top of the CURRENT integration HEAD (the original diff travels along
as attempt context). The LLM is never handed the git lifecycle.

Every merge is journaled (GIT_INTEGRATION intent -> merge -> result) and
the merge commit carries Harness-Task / Harness-Operation-Id trailers,
so crash recovery can reconcile by trailer instead of merging twice.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import StrEnum

from ..database.operation_repository import (
    OperationRepository,
    OperationStatus,
    OperationType,
)
from .checkpoint import OPERATION_TRAILER, TASK_TRAILER
from .repository import GitRepository, MergeConflict

logger = logging.getLogger(__name__)


class IntegrationStatus(StrEnum):
    MERGED = "MERGED"
    CONFLICT = "CONFLICT"
    NOOP = "NOOP"          # task branch contained no commit to integrate


@dataclass
class IntegrationOutcome:
    status: IntegrationStatus
    integration_commit: str | None = None
    operation_id: str | None = None
    conflict_files: list[str] = field(default_factory=list)
    detail: str = ""


class IntegrationManager:
    def __init__(
        self,
        repo: GitRepository,
        operations: OperationRepository | None = None,
        integration_branch: str | None = None,
    ):
        self.repo = repo                       # main checkout = integration branch
        self.operations = operations
        self.integration_branch = integration_branch
        self._lock = asyncio.Lock()            # max_git_integrations = 1, enforced
        # Like CheckpointManager: the successful merge's operation result is
        # recorded by the CALLER in the same transaction as the task-state
        # updates, so the journal and the task state settle atomically.
        self.pending_operation_id: str | None = None

    async def integrate(
        self,
        *,
        task_key: str,
        task_commit: str,
        title: str,
        project_id: int | None = None,
        task_id: int | None = None,
        attempt_id: int | None = None,
    ) -> IntegrationOutcome:
        """Merge one passed task branch into the integration branch.

        Serialized process-wide; called with the task_commit created on the
        task branch after Reviewer PASS.
        """
        async with self._lock:
            # The merge itself is subprocess work; run off the event loop so
            # parallel tasks keep their agents moving meanwhile.
            return await asyncio.to_thread(
                self._integrate_sync, task_key, task_commit, title,
                project_id, task_id, attempt_id,
            )

    def _integrate_sync(
        self,
        task_key: str,
        task_commit: str,
        title: str,
        project_id: int | None,
        task_id: int | None,
        attempt_id: int | None,
    ) -> IntegrationOutcome:
        self.pending_operation_id = None
        if self.integration_branch and self.repo.current_branch() != self.integration_branch:
            raise RuntimeError(
                f"integration checkout is on '{self.repo.current_branch()}', expected "
                f"'{self.integration_branch}' — refusing to merge into the wrong branch"
            )
        if self.repo.is_dirty():
            raise RuntimeError(
                "integration checkout is dirty; refusing to merge on top of unknown state"
            )
        base_head = self.repo.head_commit()
        if not self.repo.commit_exists(task_commit):
            raise RuntimeError(f"task commit {task_commit} does not exist")
        if self._already_contains(task_commit):
            return IntegrationOutcome(
                status=IntegrationStatus.NOOP,
                integration_commit=base_head,
                detail="task commit already reachable from integration HEAD",
            )

        operation_id = None
        if self.operations is not None:
            operation_id = self.operations.record_intent(
                OperationType.GIT_INTEGRATION,
                {
                    "task_key": task_key,
                    "task_commit": task_commit,
                    "base_head": base_head,
                    "title": title,
                },
                project_id=project_id,
                task_id=task_id,
                attempt_id=attempt_id,
            )

        message = f"integrate({task_key}): {title}"
        trailers = {TASK_TRAILER: task_key}
        if operation_id:
            trailers[OPERATION_TRAILER] = operation_id
        try:
            merge_commit = self.repo.merge_no_ff(task_commit, message, trailers=trailers)
        except MergeConflict as exc:
            conflict_files = self.repo.conflicted_paths()
            self.repo.merge_abort()
            # Conflict is a settled, terminal outcome for this operation —
            # record it immediately (no partial merge exists to reconcile).
            if operation_id and self.operations is not None:
                self.operations.record_result(
                    operation_id,
                    OperationStatus.FAILED,
                    {"conflict": True, "files": conflict_files, "detail": str(exc)},
                )
            logger.info("integration conflict on %s: %s", task_key, conflict_files)
            return IntegrationOutcome(
                status=IntegrationStatus.CONFLICT,
                operation_id=operation_id,
                conflict_files=conflict_files,
                detail=str(exc),
            )
        self.pending_operation_id = operation_id
        return IntegrationOutcome(
            status=IntegrationStatus.MERGED,
            integration_commit=merge_commit,
            operation_id=operation_id,
        )

    def _already_contains(self, commit: str) -> bool:
        result = self.repo._run("merge-base", "--is-ancestor", commit, "HEAD", check=False)
        return result.returncode == 0

    def find_integrated_operation(self, operation_id: str) -> str | None:
        """The merge commit a journaled GIT_INTEGRATION intent produced, if any."""
        return self.repo.find_commit_by_trailer(OPERATION_TRAILER, operation_id)
