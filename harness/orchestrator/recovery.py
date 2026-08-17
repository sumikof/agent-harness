"""Startup recovery.

The harness process is expected to die mid-run. On startup we reconcile
SQLite state with the Git working tree: any RUNNING attempt is treated as
interrupted, its dirty diff is archived as an artifact, the tree is reset
to the last good commit, and the task is re-queued for a fresh attempt.

A dirty tree WITHOUT a recorded RUNNING attempt is not the harness's work
to destroy: recovery refuses to start instead of resetting it.
"""

from __future__ import annotations

import logging
import sqlite3

from ..artifacts.manager import ArtifactManager
from ..database.event_repository import EventRepository
from ..database.task_repository import TaskRepository
from ..git.checkpoint import CheckpointManager
from ..git.repository import GitRepository
from .state_machine import AttemptState, TaskState

logger = logging.getLogger(__name__)

# Task states that only exist while an episode is actively running.
IN_FLIGHT_STATES = {
    TaskState.ANALYZING.value,
    TaskState.EXECUTING.value,
    TaskState.TESTING.value,
    TaskState.VERIFYING.value,
    TaskState.REVIEWING.value,
    TaskState.REPAIR_REQUIRED.value,
    TaskState.DIAGNOSING.value,
}


class UnexplainedDirtyWorktree(Exception):
    """The repository has uncommitted changes the harness did not create."""


class RecoveryManager:
    def __init__(
        self,
        tasks: TaskRepository,
        events: EventRepository,
        artifacts: ArtifactManager,
        git: GitRepository,
        checkpoint: CheckpointManager,
    ):
        self.tasks = tasks
        self.events = events
        self.artifacts = artifacts
        self.git = git
        self.checkpoint = checkpoint

    def recover(self, project_row: sqlite3.Row) -> bool:
        """Reconcile state after a possible crash. Returns True if recovery acted."""
        project_id = project_row["id"]
        acted = False

        running = self.tasks.running_attempts(project_id)
        dirty = self.git.is_repo() and self.git.is_dirty()

        if running:
            logger.warning(
                "recovery: %d running attempt(s), working tree dirty=%s", len(running), dirty
            )
            self._archive_dirty_diff(project_id)
            for attempt in running:
                self.tasks.finish_attempt(attempt["id"], AttemptState.INTERRUPTED)
                self.events.emit(
                    "ATTEMPT_INTERRUPTED",
                    project_id=project_id,
                    task_id=attempt["task_id"],
                    attempt_id=attempt["id"],
                    payload={"recovered_at_startup": True},
                )
            if dirty:
                self.checkpoint.discard_working_tree()
            acted = True
        elif dirty:
            # No interrupted attempt explains this diff — it is user work.
            # Never destroy it; make the operator decide.
            raise UnexplainedDirtyWorktree(
                f"repository {self.git.path} has uncommitted changes but no interrupted "
                "attempt is recorded. Commit, stash, or clean it manually, then rerun."
            )

        for task in self.tasks.list_for_project(project_id):
            if task["status"] in IN_FLIGHT_STATES:
                self.tasks.set_status(task["id"], TaskState.READY, force=True)
                self.events.emit(
                    "TASK_REQUEUED",
                    project_id=project_id,
                    task_id=task["id"],
                    payload={"from_status": task["status"]},
                )
                acted = True

        if acted:
            self.events.emit("RECOVERY_COMPLETED", project_id=project_id,
                             payload={"head": self.git.head_commit() if self.git.is_repo() else None})
        return acted

    def _archive_dirty_diff(self, project_id: int) -> None:
        if not (self.git.is_repo() and self.git.is_dirty()):
            return
        try:
            diff = self.git.full_dirty_diff()
        except Exception as exc:
            logger.warning("recovery: could not capture dirty diff: %s", exc)
            return
        if diff.strip():
            path = self.artifacts.root / "diagnostics" / "interrupted-worktree.diff"
            # Keep prior archives; suffix with the event id ordering via count
            index = 0
            while path.exists():
                index += 1
                path = self.artifacts.root / "diagnostics" / f"interrupted-worktree-{index}.diff"
            self.artifacts.save_text(path, diff)
            logger.info("recovery: archived interrupted diff to %s", path)
