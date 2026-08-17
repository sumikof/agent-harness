"""Startup recovery.

The harness process is expected to die mid-run. On startup, state is
reconciled in a fixed order:

    1. SQLite integrity check
    2. Event ledger integrity check (contiguous per-stream seq)
    3. Unfinished operation detection (intent without result)
    4. Git reconciliation (did a journaled commit actually happen?)
    5. Workspace dirty-state recovery (archive diff, reset to HEAD)
    6. Project / task state reconciliation (requeue in-flight work)

A journaled GIT_COMMIT intent whose commit already exists in the
repository is never re-executed: the DB is reconciled to the real git
state instead. Only after operations are settled may the working tree be
reset — never unconditionally.

A dirty tree WITHOUT a recorded RUNNING attempt is not the harness's work
to destroy: recovery refuses to start instead of resetting it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

from ..agents import analyst, developer, diagnostician, planner, reviewer, tester
from ..artifacts.manager import ArtifactManager
from ..database.event_repository import EventRepository, EventType
from ..database.operation_repository import (
    OperationRepository,
    OperationStatus,
    OperationType,
)
from ..database.run_repository import RunRepository
from ..database.task_repository import TaskRepository
from ..git.checkpoint import CheckpointManager
from ..git.repository import GitRepository
from .state_machine import AttemptState, TaskState

logger = logging.getLogger(__name__)

# Roles whose sessions may write to the worktree (from the role specs) —
# only their in-flight dispatches can explain a dirty tree.
MUTATING_ROLES = {
    spec.role.value
    for spec in (planner.SPEC, analyst.SPEC, developer.SPEC,
                 tester.SPEC, reviewer.SPEC, diagnostician.SPEC)
    if spec.mutates_repo
}

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


class RecoveryIntegrityError(Exception):
    """The persistent state plane itself cannot be trusted (corrupt SQLite
    file or a gapped event stream). Never silently repaired."""


class RecoveryManager:
    def __init__(
        self,
        tasks: TaskRepository,
        events: EventRepository,
        artifacts: ArtifactManager,
        git: GitRepository,
        checkpoint: CheckpointManager,
        operations: OperationRepository | None = None,
        runs: RunRepository | None = None,
    ):
        self.tasks = tasks
        self.events = events
        self.artifacts = artifacts
        self.git = git
        self.checkpoint = checkpoint
        self.operations = operations
        self.runs = runs

    def recover(self, project_row: sqlite3.Row) -> bool:
        """Reconcile state after a possible crash. Returns True if recovery acted."""
        project_id = project_row["id"]
        acted = False

        # 1-2. Integrity of the state plane itself. Corruption is fatal and
        # loud — recovery must not guess on top of untrusted state.
        integrity = self.tasks.db.integrity_check()
        if integrity != "ok":
            raise RecoveryIntegrityError(f"SQLite integrity check failed: {integrity}")
        gaps = self.events.find_stream_gaps()
        if gaps:
            raise RecoveryIntegrityError(
                f"event ledger has non-contiguous streams: {gaps}. The history "
                "cannot be trusted; inspect the workspace before rerunning."
            )

        # 3-4. Unfinished operations. Git commits are reconciled BEFORE any
        # tree reset; in-flight dispatch/verification intents are only READ
        # here — they stay open until the tree is settled below, so a crash
        # inside recovery itself keeps the evidence for the next startup.
        pending_side_effects = 0
        unexecuted_commit_intents: list[sqlite3.Row] = []
        if self.operations is not None:
            for op in self.operations.unfinished(OperationType.GIT_COMMIT, project_id=project_id):
                commit_hash = self.checkpoint.find_committed_operation(op["operation_id"])
                if commit_hash is not None:
                    # The commit exists — reconcile the DB now; this touches
                    # no worktree state and must precede any reset.
                    self._reconcile_executed_commit(project_id, op, commit_hash)
                    acted = True
                else:
                    # Intent journaled but never executed. It stays PENDING
                    # until the reset below has completed (double-crash
                    # safety), and is closed FAILED afterwards.
                    unexecuted_commit_intents.append(op)
            # In-flight side effects that can explain a dirty tree: a
            # verification command, an unexecuted checkpoint commit, or a
            # MUTATING agent session whose attempt was already closed by a
            # previous recovery pass that crashed mid-reset. A read-only
            # dispatch (e.g. the Planner, which runs without an attempt)
            # cannot have produced the diff — counting it would let user
            # edits made while stopped be destroyed.
            pending_side_effects = len(self.operations.unfinished(
                OperationType.VERIFICATION_COMMAND, project_id=project_id))
            # An unexecuted commit intent is evidence ONLY when the current
            # dirty diff still hashes to the intent's recorded diff_sha256.
            # A stale intent (its reset already completed, then the process
            # died before closing it) must not explain — and destroy — NEW
            # user edits made while the harness was stopped.
            pending_side_effects += sum(
                1 for op in unexecuted_commit_intents
                if self._commit_intent_matches_worktree(op)
            )
            for op in self.operations.unfinished(
                    OperationType.AGENT_DISPATCH, project_id=project_id):
                payload = json.loads(op["payload"] or "{}")
                if op["attempt_id"] is not None and payload.get("role") in MUTATING_ROLES:
                    pending_side_effects += 1

        # 5. Workspace dirty-state recovery.
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
                    EventType.ATTEMPT_INTERRUPTED,
                    project_id=project_id,
                    task_id=attempt["task_id"],
                    attempt_id=attempt["id"],
                    payload={"recovered_at_startup": True},
                )
            if dirty:
                self.checkpoint.discard_working_tree()
            acted = True
        elif dirty and pending_side_effects:
            # No RUNNING attempt, but a journaled side effect was in flight —
            # a verification command (e.g. the final verification, which runs
            # outside any attempt) or an agent dispatch whose attempt a
            # crashed earlier recovery already closed. Either explains the
            # diff: archive it and return to the last good commit.
            logger.warning(
                "recovery: dirty tree explained by %d in-flight side-effect "
                "operation(s); archiving and resetting", pending_side_effects,
            )
            self._archive_dirty_diff(project_id)
            self.checkpoint.discard_working_tree()
            acted = True
        elif dirty:
            # No interrupted attempt explains this diff — it is user work.
            # Never destroy it; make the operator decide.
            raise UnexplainedDirtyWorktree(
                f"repository {self.git.path} has uncommitted changes but no interrupted "
                "attempt is recorded. Commit, stash, or clean it manually, then rerun."
            )

        # Now that the tree is settled, the in-flight operations can be
        # closed. Crash-before-this-point keeps them PENDING, so the next
        # startup still sees the evidence and repeats the steps above.
        if self.operations is not None:
            for op in unexecuted_commit_intents:
                self.operations.record_result(
                    op["operation_id"], OperationStatus.FAILED,
                    {"reason": "intent journaled but commit never executed"},
                )
                logger.warning(
                    "recovery: git commit intent %s never executed", op["operation_id"]
                )
                acted = True
            acted |= self._close_interrupted_operations(project_id)

        # 6. Project / task state reconciliation.
        if self.runs is not None:
            interrupted_runs = self.runs.interrupt_running(project_id)
            if interrupted_runs:
                logger.warning("recovery: closed %d interrupted agent run(s)", len(interrupted_runs))
                acted = True
        for task in self.tasks.list_for_project(project_id):
            if task["status"] in IN_FLIGHT_STATES:
                self.tasks.set_status(task["id"], TaskState.READY, force=True)
                self.events.emit(
                    EventType.TASK_REQUEUED,
                    project_id=project_id,
                    task_id=task["id"],
                    payload={"from_status": task["status"]},
                )
                acted = True

        if acted:
            self.events.emit(EventType.RECOVERY_COMPLETED, project_id=project_id,
                             payload={"head": self.git.head_commit() if self.git.is_repo() else None})
        return acted

    # ------------------------------------------------------------------

    def _commit_intent_matches_worktree(self, op: sqlite3.Row) -> bool:
        """True when the current dirty diff is exactly what the commit intent
        was journaled for (payload diff_sha256). Unverifiable intents are
        never accepted as grounds to reset a tree."""
        payload = json.loads(op["payload"] or "{}")
        expected = payload.get("diff_sha256")
        if not expected:
            return False
        if not (self.git.is_repo() and self.git.is_dirty()):
            return False
        try:
            diff = self.git.full_dirty_diff()
        except Exception as exc:
            logger.warning("recovery: could not hash dirty diff: %s", exc)
            return False
        return hashlib.sha256(diff.encode("utf-8")).hexdigest() == expected

    def _close_interrupted_operations(self, project_id: int) -> bool:
        """Close in-flight dispatch/verification intents AFTER the worktree
        has been settled. Scoped to THIS project: other projects in the same
        workspace keep their pending journals for their own startup."""
        acted = False
        for op_type in (OperationType.AGENT_DISPATCH, OperationType.VERIFICATION_COMMAND):
            for op in self.operations.unfinished(op_type, project_id=project_id):
                # The side effect (an agent session / a verification process)
                # died with the harness; its worktree effects were handled by
                # the dirty-state step, so the operation is closed as
                # interrupted rather than guessed at.
                self.operations.record_result(
                    op["operation_id"],
                    OperationStatus.INTERRUPTED,
                    {"reason": "harness crashed while operation was in flight"},
                )
                acted = True
        return acted

    def _reconcile_executed_commit(
        self, project_id: int, op: sqlite3.Row, commit_hash: str
    ) -> None:
        """A GIT_COMMIT intent whose commit exists in the repository: the
        side effect is real — the DB is caught up to it and the commit is
        NEVER re-run. (Unexecuted intents are handled by the caller: kept
        PENDING as dirty-tree evidence until the reset completed, then
        closed FAILED so normal retry produces a fresh attempt.)
        """
        operation_id = op["operation_id"]
        payload = json.loads(op["payload"] or "{}")
        task_id = payload.get("task_id") or op["task_id"]
        attempt_id = payload.get("attempt_id") or op["attempt_id"]

        logger.warning(
            "recovery: git commit intent %s already executed as %s; reconciling DB",
            operation_id, commit_hash,
        )
        # The whole catch-up — attempt, current_commit, task state, the
        # operation result and every event — lands in ONE transaction,
        # mirroring the normal completion path. A crash mid-reconciliation
        # then leaves the operation PENDING and the next startup repeats
        # the reconciliation from scratch, instead of a permanent
        # state/ledger mismatch.
        with self.tasks.db.transaction():
            if task_id is not None:
                attempt = self.tasks.get_attempt(attempt_id) if attempt_id else None
                if attempt is not None and attempt["status"] == AttemptState.RUNNING.value:
                    self.tasks.finish_attempt(attempt_id, AttemptState.PASSED)
                self.tasks.set_commit(task_id, commit_hash)
                task = self.tasks.get(task_id)
                if task is not None and task["status"] != TaskState.COMPLETED.value:
                    self.tasks.set_status(task_id, TaskState.COMPLETED, force=True)
                    self.events.emit(
                        EventType.TASK_COMPLETED,
                        project_id=project_id,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        operation_id=operation_id,
                        payload={"commit": commit_hash, "reconciled": True},
                    )
            self.operations.record_result(
                operation_id, OperationStatus.RECONCILED, {"commit": commit_hash}
            )
            self.events.emit(
                EventType.GIT_COMMIT_RECONCILED,
                project_id=project_id,
                task_id=task_id,
                attempt_id=attempt_id,
                operation_id=operation_id,
                payload={"commit": commit_hash},
            )

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
