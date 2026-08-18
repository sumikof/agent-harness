"""Startup recovery.

The harness process is expected to die mid-run — possibly with MANY
tasks RUNNING in parallel, each in its own worktree. On startup, state
is reconciled in a fixed order:

    1. SQLite integrity check
    2. Event ledger integrity check (contiguous per-stream seq)
    3. Unfinished operation detection (intent without result)
    4. Git reconciliation:
       - GIT_COMMIT intents (task-branch commits) by trailer
       - GIT_INTEGRATION intents by trailer; an in-progress merge in the
         integration checkout is aborted (never finished blindly, never
         merged twice — the Operation ID decides)
    5. Worktree recovery, PER RUNNING ATTEMPT: archive that worktree's
       dirty diff, remove the worktree, close the attempt INTERRUPTED.
       Other tasks' worktrees are never touched. Orphan worktrees (no
       RUNNING attempt) are archived and removed too.
    6. Main-checkout dirty-state recovery (evidence-hash gated, as before)
    7. Project / task state reconciliation (requeue ALL in-flight tasks)

A journaled intent whose side effect already exists in git is never
re-executed: the DB is reconciled to the real git state instead.

A dirty INTEGRATION checkout without journaled evidence is not the
harness's work to destroy: recovery refuses to start instead of
resetting it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

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
    TaskState.INTEGRATING.value,
    TaskState.INTEGRATION_CONFLICT.value,
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
        worktrees=None,
        integration=None,
    ):
        self.tasks = tasks
        self.events = events
        self.artifacts = artifacts
        self.git = git
        self.checkpoint = checkpoint
        self.operations = operations
        self.runs = runs
        self.worktrees = worktrees        # WorktreeManager | None
        self.integration = integration    # IntegrationManager | None

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

        # 3a. Integration reconciliation FIRST: a crash mid-merge leaves the
        # integration checkout in a merge state that must be aborted (or the
        # DB caught up to an already-executed merge) BEFORE the dirty-state
        # evidence rules below inspect the tree. Operation IDs decide —
        # a merge is never executed twice.
        if self.operations is not None:
            acted |= self._reconcile_integrations(project_id)

        # 3-4. Unfinished operations. Git commits are reconciled BEFORE any
        # tree reset; in-flight dispatch/verification intents are only READ
        # here — they stay open until the tree is settled below, so a crash
        # inside recovery itself keeps the evidence for the next startup.
        dirty = self.git.is_repo() and self.git.is_dirty()
        # Computed WITHOUT touching the user's index (no intent-to-add
        # residue) — the tree may turn out to be user work we must not alter.
        current_diff_hash = self._current_diff_hash() if dirty else None

        pending_side_effects = 0
        unexecuted_commit_intents: list[sqlite3.Row] = []
        side_effect_intents: list[sqlite3.Row] = []
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
            # In-flight side effects that can explain a dirty MAIN checkout:
            # a verification command, an unexecuted checkpoint commit, or a
            # MUTATING agent session whose attempt was already closed by a
            # previous recovery pass that crashed mid-reset. A read-only
            # dispatch (e.g. the Planner, which runs without an attempt)
            # cannot have produced the diff — counting it would let user
            # edits made while stopped be destroyed. Intents that ran inside
            # a task WORKTREE are excluded here: their effects are settled
            # by the per-worktree recovery below, never against the main
            # checkout.
            side_effect_intents = [
                op for op in self.operations.unfinished(
                    OperationType.VERIFICATION_COMMAND, project_id=project_id)
                if self._is_main_checkout_intent(op)
            ]
            for op in self.operations.unfinished(
                    OperationType.AGENT_DISPATCH, project_id=project_id):
                payload = json.loads(op["payload"] or "{}")
                if (op["attempt_id"] is not None and payload.get("role") in MUTATING_ROLES
                        and self._is_main_checkout_intent(op)):
                    side_effect_intents.append(op)
            # Staleness rules: a commit intent is evidence only when the
            # current diff hashes to its recorded diff_sha256. A dispatch/
            # verification intent that already went through a settlement pass
            # (annotated below with the diff it settled) is evidence only
            # when the diff is still that one — otherwise the tree holds NEW
            # user edits made while the harness was stopped, which must be
            # preserved, not archived and reset.
            pending_side_effects = sum(
                1 for op in side_effect_intents
                if self._intent_is_dirty_evidence(op, current_diff_hash)
            ) + sum(
                1 for op in unexecuted_commit_intents
                if self._commit_intent_matches(op, current_diff_hash)
            )

        # 5. Worktree recovery — PER RUNNING ATTEMPT. Every parallel task
        # that was in flight when the process died is recovered on its own:
        # its worktree diff archived, its worktree removed, its attempt
        # closed. No other task's worktree is touched. Attempts without a
        # recorded worktree (pre-worktree data) fall through to the main-
        # checkout recovery below.
        running = self.tasks.running_attempts(project_id)
        worktree_attempts = [a for a in running if a["worktree_path"]]
        legacy_attempts = [a for a in running if not a["worktree_path"]]
        for attempt in worktree_attempts:
            self._recover_worktree_attempt(project_id, attempt)
            acted = True
        acted |= self._recover_orphan_worktrees(project_id)

        # 6. Main-checkout dirty-state recovery (evidence-hash gated).
        if legacy_attempts:
            logger.warning(
                "recovery: %d running attempt(s) on the main checkout, dirty=%s",
                len(legacy_attempts), dirty,
            )
            # Durable BEFORE the reset: a crash after the reset but before
            # the intents are closed leaves them annotated with the settled
            # diff, so the next startup can tell them apart from new work.
            self._annotate_settlement(side_effect_intents, current_diff_hash)
            self._archive_dirty_diff(project_id)
            for attempt in legacy_attempts:
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
            self._annotate_settlement(side_effect_intents, current_diff_hash)
            self._archive_dirty_diff(project_id)
            self.checkpoint.discard_working_tree()
            acted = True
        elif dirty:
            # No verifiable explanation for this diff — it may be user work.
            # Never destroy it; make the operator decide.
            hint = ""
            if side_effect_intents or unexecuted_commit_intents:
                hint = (
                    " In-flight operation(s) were interrupted, but the current diff "
                    "does not match any state the harness recorded for them, so it "
                    "may include your own edits."
                )
            raise UnexplainedDirtyWorktree(
                f"repository {self.git.path} has uncommitted changes but no interrupted "
                f"attempt is recorded.{hint} Commit, stash, or clean it manually, then rerun."
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

        # 6. Project / task state reconciliation. RUNNING run rows are closed
        # WORKSPACE-WIDE: the workspace lock guarantees no other live
        # process, and a stale row from any project would block the shared
        # max-1-agent slot for every project in this DB.
        if self.runs is not None:
            interrupted_runs = self.runs.interrupt_running()
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

    def _is_main_checkout_intent(self, op: sqlite3.Row) -> bool:
        """True when this intent's side effects landed in the MAIN checkout
        (final verification, pre-worktree data) rather than a task worktree."""
        payload = json.loads(op["payload"] or "{}")
        if payload.get("worktree"):
            return False
        if op["attempt_id"] is not None:
            attempt = self.tasks.get_attempt(op["attempt_id"])
            if attempt is not None and attempt["worktree_path"]:
                return False
        return True

    def _recover_worktree_attempt(self, project_id: int, attempt: sqlite3.Row) -> None:
        """Settle ONE crashed task attempt: archive its worktree's dirty
        diff, remove the worktree + branch, close the attempt INTERRUPTED.

        The worktree is harness-owned for the episode's whole lifetime, so
        (like the v1 RUNNING-attempt rule) it is archived and removed
        without evidence hashing — the diff is never destroyed, always
        archived first.
        """
        worktree_path = attempt["worktree_path"]
        branch = attempt["branch"]
        path = Path(worktree_path)
        if path.is_dir():
            repo = GitRepository(path)
            try:
                if repo.is_repo() and repo.is_dirty():
                    diff = repo.snapshot_dirty_bytes()
                    archive = (self.artifacts.root / "diagnostics"
                               / f"interrupted-{path.name}.diff")
                    index = 0
                    while archive.exists():
                        index += 1
                        archive = (self.artifacts.root / "diagnostics"
                                   / f"interrupted-{path.name}-{index}.diff")
                    if diff.strip():
                        self.artifacts.save_bytes(archive, diff)
                    self.artifacts.archive_worktree_files(
                        archive.with_suffix(".files.tar"), repo.path, repo.changed_paths()
                    )
                    logger.info("recovery: archived worktree %s to %s", path, archive)
            except Exception as exc:
                raise RecoveryIntegrityError(
                    f"could not archive interrupted worktree {path} before removal: {exc}"
                )
            if self.worktrees is not None:
                self.worktrees.remove_path(path, branch=branch)
        elif branch and self.worktrees is not None:
            self.worktrees.remove_path(path, branch=branch)
        self.tasks.finish_attempt(attempt["id"], AttemptState.INTERRUPTED)
        self.events.emit(
            EventType.ATTEMPT_INTERRUPTED,
            project_id=project_id,
            task_id=attempt["task_id"],
            attempt_id=attempt["id"],
            payload={"recovered_at_startup": True, "worktree": worktree_path},
        )
        logger.warning("recovery: interrupted attempt %d (worktree %s)",
                       attempt["id"], worktree_path)

    def _recover_orphan_worktrees(self, project_id: int) -> bool:
        """HARNESS-OWNED worktrees no RUNNING attempt references are crash
        leftovers (their attempt settled in an earlier partial recovery):
        archive whatever they hold and remove them.

        Scope is deliberately narrow. A repository may carry worktrees the
        harness never created — the operator's own checkouts — and those are
        not recovery's to destroy: force-removing one and `branch -D`-ing its
        branch can drop the only reference to unmerged commits. Only
        worktrees under this manager's root, on a branch under the harness
        branch prefix, are treated as orphans.
        """
        if self.worktrees is None or not self.git.is_repo():
            return False
        referenced = {
            str(Path(a["worktree_path"]).resolve())
            for a in self.tasks.running_attempts(project_id)
            if a["worktree_path"]
        }
        acted = False
        for path in self.worktrees.managed_paths():
            if str(path.resolve()) in referenced:
                continue
            checked_out = None
            if path.is_dir():
                candidate = GitRepository(path)
                if candidate.is_repo():
                    checked_out = candidate.current_branch()
            # A worktree inside our root but on a foreign branch is not ours
            # to remove — leave it and let the operator decide.
            if checked_out is not None and not self.worktrees.owns_checkout(checked_out):
                logger.warning(
                    "recovery: leaving worktree %s alone; branch '%s' is not a "
                    "harness task branch", path, checked_out,
                )
                continue
            repo = GitRepository(path)
            try:
                if path.is_dir() and repo.is_repo() and repo.is_dirty():
                    diff = repo.snapshot_dirty_bytes()
                    archive = (self.artifacts.root / "diagnostics"
                               / f"orphan-{path.name}.diff")
                    if diff.strip():
                        self.artifacts.save_bytes(archive, diff)
                    self.artifacts.archive_worktree_files(
                        archive.with_suffix(".files.tar"), repo.path, repo.changed_paths()
                    )
            except Exception as exc:
                raise RecoveryIntegrityError(
                    f"could not archive orphan worktree {path} before removal: {exc}"
                )
            # Only a harness task branch is ever deleted along with the tree.
            branch = checked_out if self.worktrees.owns_branch(checked_out) else None
            self.worktrees.remove_path(path, branch=branch)
            logger.warning("recovery: removed orphan worktree %s (branch %s)", path, branch)
            acted = True
        return acted

    def _reconcile_integrations(self, project_id: int) -> bool:
        """Settle unfinished GIT_INTEGRATION intents against real git state.

        Executed merge (found by Operation-Id trailer) -> catch the DB up
        (task COMPLETED with integration_commit); never merged -> abort any
        in-progress merge and fail the intent (the task requeues normally).
        A merge is never executed twice.
        """
        acted = False
        for op in self.operations.unfinished(OperationType.GIT_INTEGRATION,
                                             project_id=project_id):
            merge_commit = self.checkpoint.find_committed_operation(op["operation_id"])
            if merge_commit is not None:
                self._reconcile_executed_integration(project_id, op, merge_commit)
            else:
                if self.git.is_repo() and self.git.merge_in_progress():
                    logger.warning(
                        "recovery: aborting in-progress merge for integration intent %s",
                        op["operation_id"],
                    )
                    self.git.merge_abort()
                self.operations.record_result(
                    op["operation_id"], OperationStatus.FAILED,
                    {"reason": "integration intent journaled but merge never completed"},
                )
            acted = True
        return acted

    def _reconcile_executed_integration(
        self, project_id: int, op: sqlite3.Row, merge_commit: str
    ) -> None:
        payload = json.loads(op["payload"] or "{}")
        task_id = op["task_id"]
        attempt_id = op["attempt_id"]
        logger.warning(
            "recovery: integration intent %s already merged as %s; reconciling DB",
            op["operation_id"], merge_commit,
        )
        with self.tasks.db.transaction():
            if task_id is not None:
                self.tasks.set_integration_commit(task_id, merge_commit)
                task = self.tasks.get(task_id)
                if task is not None and task["status"] != TaskState.COMPLETED.value:
                    self.tasks.set_status(task_id, TaskState.COMPLETED, force=True)
                    self.events.emit(
                        EventType.TASK_COMPLETED,
                        project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                        operation_id=op["operation_id"],
                        payload={"commit": merge_commit, "reconciled": True},
                    )
            self.operations.record_result(
                op["operation_id"], OperationStatus.RECONCILED,
                {"integration_commit": merge_commit},
            )
            self.events.emit(
                EventType.GIT_INTEGRATION_RECONCILED,
                project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                operation_id=op["operation_id"],
                payload={"integration_commit": merge_commit,
                         "task_commit": payload.get("task_commit")},
            )

    def _current_diff_hash(self) -> str | None:
        try:
            # Fingerprint of the ACTUAL on-disk bytes — git filters cannot
            # make two different worktree states hash alike.
            return self.git.dirty_state_hash()
        except Exception as exc:
            logger.warning("recovery: could not hash dirty state: %s", exc)
            return None

    @staticmethod
    def _commit_intent_matches(op: sqlite3.Row, current_diff_hash: str | None) -> bool:
        """True when the current dirty diff is exactly what the commit intent
        was journaled for (payload diff_sha256). Unverifiable intents are
        never accepted as grounds to reset a tree."""
        payload = json.loads(op["payload"] or "{}")
        expected = payload.get("diff_sha256")
        return bool(expected) and expected == current_diff_hash

    @staticmethod
    def _intent_is_dirty_evidence(op: sqlite3.Row, current_diff_hash: str | None) -> bool:
        """A dispatch/verification intent explains the dirty tree only when
        the current diff hashes to a state the harness has durably recorded
        for it: the base diff journaled at intent creation, or the diff a
        previous settlement pass annotated before resetting.

        The invariant: recovery only ever resets a tree whose exact content
        it can prove it has seen. A diff matching neither hash may contain
        user edits made while the harness was stopped — never reset those.
        """
        payload = json.loads(op["payload"] or "{}")
        annotated = payload.get("settle_diff_sha256")
        if annotated is not None:
            return annotated == current_diff_hash
        base = payload.get("base_diff_sha256")
        return bool(base) and base == current_diff_hash

    def _annotate_settlement(
        self, ops: list[sqlite3.Row], current_diff_hash: str | None
    ) -> None:
        """Durably mark which dirty diff these in-flight intents are being
        settled against, before the tree is reset."""
        if current_diff_hash is None:
            return
        for op in ops:
            payload = json.loads(op["payload"] or "{}")
            if payload.get("settle_diff_sha256") == current_diff_hash:
                continue
            self.operations.annotate(
                op["operation_id"], {"settle_diff_sha256": current_diff_hash}
            )

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

        Under the worktree model a task-branch commit is NOT completion —
        completion is the serialized integration merge (handled by
        _reconcile_integrations). The attempt is settled as PASSED and the
        task_commit recorded; the task itself requeues through the normal
        in-flight reconciliation and re-runs on a fresh worktree.
        """
        operation_id = op["operation_id"]
        payload = json.loads(op["payload"] or "{}")
        task_id = payload.get("task_id") or op["task_id"]
        attempt_id = payload.get("attempt_id") or op["attempt_id"]

        logger.warning(
            "recovery: git commit intent %s already executed as %s; reconciling DB",
            operation_id, commit_hash,
        )
        # Settling the attempt makes its worktree an orphan, and orphan
        # cleanup deletes the task branch — possibly this commit's only
        # ref. Pin it FIRST (and outside the transaction: a pinned ref with
        # an unreconciled DB is harmless, the reverse loses the commit).
        self.git.pin_ref(
            f"refs/harness/reconciled/attempt-{attempt_id or operation_id}",
            commit_hash,
        )
        # The whole catch-up lands in ONE transaction. A crash
        # mid-reconciliation leaves the operation PENDING and the next
        # startup repeats the reconciliation from scratch, instead of a
        # permanent state/ledger mismatch.
        with self.tasks.db.transaction():
            if task_id is not None:
                attempt = self.tasks.get_attempt(attempt_id) if attempt_id else None
                if attempt is not None and attempt["status"] == AttemptState.RUNNING.value:
                    self.tasks.finish_attempt(attempt_id, AttemptState.PASSED)
                self.tasks.set_task_commit(task_id, commit_hash)
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
            # Byte-exact, binary-safe, re-applicable patch: the archive is
            # the only copy once the tree is reset, so nothing may be lost
            # to text decoding or diff rendering.
            diff = self.git.snapshot_dirty_bytes()
        except Exception as exc:
            # Without a verified archive there is nothing safe to reset —
            # abort recovery loudly instead of destroying unarchived work.
            raise RecoveryIntegrityError(
                f"could not archive the dirty worktree before reset: {exc}"
            )
        path = self.artifacts.root / "diagnostics" / "interrupted-worktree.diff"
        # Keep prior archives; suffix with the event id ordering via count
        index = 0
        while path.exists() or path.with_suffix(".files.tar").exists():
            index += 1
            path = self.artifacts.root / "diagnostics" / f"interrupted-worktree-{index}.diff"
        if diff.strip():
            self.artifacts.save_bytes(path, diff)
        # Ground truth REGARDLESS of the patch: a clean filter can normalize
        # the diff to empty while the on-disk bytes still differ, and the
        # reset below would destroy them. A tree that cannot be archived
        # faithfully (sockets, devices) must not be reset at all.
        try:
            self.artifacts.archive_worktree_files(
                path.with_suffix(".files.tar"), self.git.path, self.git.changed_paths()
            )
        except Exception as exc:
            raise RecoveryIntegrityError(
                f"could not archive the dirty worktree before reset: {exc}"
            )
        logger.info("recovery: archived interrupted worktree to %s(.files.tar)", path)
