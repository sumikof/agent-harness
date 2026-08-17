"""Task Loop: Analyst -> Developer -> Tester -> Verification -> Reviewer.

All routing decisions here are deterministic Python. Agents think;
the harness decides who runs next.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from enum import StrEnum

from ..agents import analyst, developer, diagnostician, reviewer, tester
from ..artifacts.manager import ArtifactManager
from ..artifacts.schemas import Diagnosis, Review, VerificationResult
from ..config import HarnessConfig
from ..context.attempt_context import AttemptContext
from ..context.project_context import ProjectContext
from ..context.task_context import TaskContext
from ..database.event_repository import EventRepository
from ..database.operation_repository import (
    OperationRepository,
    OperationStatus,
    OperationType,
)
from ..database.run_repository import VERIFY_FAIL, VERIFY_PASS, RunRepository
from ..database.task_repository import TaskRepository
from ..git.checkpoint import CheckpointManager
from ..git.repository import GitRepository
from ..verification.runner import VerificationRunner
from .agent_invoker import AgentConfigurationError, AgentInvoker, AgentRunFailed
from .budget import BudgetExceeded
from .state_machine import (
    AttemptState,
    DiagnosisVerdict,
    ReviewVerdict,
    Role,
    TaskState,
    decide_after_review,
    decide_after_verification,
)

logger = logging.getLogger(__name__)

MAX_DIAGNOSIS_ROUNDS = 2
REVIEW_DIFF_LIMIT = 30000


class TaskOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    REPLAN = "REPLAN"
    SPLIT = "SPLIT"
    BLOCKED = "BLOCKED"


class TaskRunner:
    def __init__(
        self,
        config: HarnessConfig,
        invoker: AgentInvoker,
        tasks: TaskRepository,
        runs: RunRepository,
        events: EventRepository,
        artifacts: ArtifactManager,
        git: GitRepository,
        checkpoint: CheckpointManager,
        verifier: VerificationRunner,
        operations: OperationRepository | None = None,
    ):
        self.config = config
        self.invoker = invoker
        self.tasks = tasks
        self.runs = runs
        self.events = events
        self.artifacts = artifacts
        self.git = git
        self.checkpoint = checkpoint
        self.verifier = verifier
        self.operations = operations

    async def run_task(
        self, project_id: int, task_row: sqlite3.Row, project_ctx: ProjectContext
    ) -> TaskOutcome:
        task_id = task_row["id"]
        task_key = task_row["task_key"]
        if task_row["status"] == TaskState.PENDING.value:
            self.tasks.set_status(task_id, TaskState.READY)
        self.events.emit("TASK_STARTED", project_id=project_id, task_id=task_id,
                         payload={"task_key": task_key})

        task_ctx = self._build_task_ctx(task_row)
        feedback = AttemptContext()
        diagnosis_rounds = 0
        need_analysis = True

        while True:
            try:
                # The attempt is opened BEFORE analysis so that an Analyst
                # failure is recorded as a failed attempt and counts toward
                # the retry limit — otherwise a failing analysis would loop
                # outside every budget except wall clock.
                attempt_id = self.tasks.start_attempt(task_id, self.git.head_commit())
                feedback.attempt_no = self.tasks.get(task_id)["attempt_count"]

                if need_analysis:
                    self.tasks.set_status(task_id, TaskState.ANALYZING, force=True)
                    brief = await self.invoker.invoke(
                        analyst.SPEC,
                        project_id,
                        project_ctx,
                        task_ctx=task_ctx,
                        attempt_ctx=feedback,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        artifact_path=self.artifacts.task_artifact_path(task_key, "task-brief.json"),
                    )
                    task_ctx.task_brief = brief.model_dump()
                    task_ctx.relevant_files = list(brief.files)
                    need_analysis = False

                kind, payload = await self._run_attempt(
                    project_id, task_id, task_key, attempt_id, project_ctx, task_ctx, feedback
                )
            except BudgetExceeded as exc:
                # A pause must leave the repo at the last good commit —
                # otherwise the next startup finds a dirty tree with no
                # RUNNING attempt and refuses to start. The diff is archived
                # first so nothing is lost.
                self._archive_and_discard(task_key, "budget-paused")
                self._finish_running_attempts(task_id)
                if exc.scope == "task":
                    self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
                    self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                                     payload={"reason": str(exc)})
                    return TaskOutcome.BLOCKED
                raise  # project-scope budget: the Project Loop pauses the project
            except AgentConfigurationError as exc:
                # A provider/profile misconfiguration fails identically on
                # every retry: block loudly instead of burning attempts.
                logger.error("configuration error on %s: %s", task_key, exc)
                self._archive_and_discard(task_key, "config-error")
                self._finish_running_attempts(task_id, AttemptState.FAILED)
                self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
                self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                                 payload={"reason": f"configuration error: {exc.detail}"})
                return TaskOutcome.BLOCKED
            except AgentRunFailed as exc:
                logger.error("agent run failed on %s: %s", task_key, exc)
                # The attempt opened above is closed as FAILED, so this
                # failure counts toward consecutive_failures().
                self._finish_running_attempts(task_id, AttemptState.FAILED)
                self.tasks.set_status(task_id, TaskState.FAILED, force=True)
                # Capture the half-done diff now: the Diagnostician needs it
                # in its context, and the tree is reset before diagnosis.
                feedback = AttemptContext(
                    previous_attempt_summary=f"Previous attempt aborted: {exc.detail}",
                    current_diff=self._bounded_diff(f"{task_key}-a{attempt_id}-aborted.diff"),
                )
                kind, payload = ("AGENT_FAILURE", None)

            # ---- deterministic routing --------------------------------------
            if kind == "PASS":
                return self._complete_task(project_id, task_id, task_key, attempt_id, task_row)

            if kind == "REPLAN":
                self.tasks.finish_attempt(attempt_id, AttemptState.FAILED)
                self.checkpoint.discard_working_tree()
                self.tasks.set_status(task_id, TaskState.PENDING, force=True)
                self.events.emit("REVIEW_REPLAN", project_id=project_id, task_id=task_id,
                                 attempt_id=attempt_id,
                                 payload={"reason": payload.replan_reason if payload else ""})
                return TaskOutcome.REPLAN

            if kind == "VERIFY_FAIL":
                self.tasks.finish_attempt(attempt_id, AttemptState.FAILED)
                self.tasks.set_status(task_id, TaskState.REPAIR_REQUIRED, force=True)
                feedback = AttemptContext(
                    verification_failure=payload.model_dump(),
                    current_diff=self._bounded_diff(f"{task_key}-a{attempt_id}-verify-fail.diff"),
                )
            elif kind == "REPAIR":
                self.tasks.finish_attempt(attempt_id, AttemptState.FAILED)
                self.tasks.set_status(task_id, TaskState.REPAIR_REQUIRED, force=True)
                feedback = AttemptContext(
                    review_feedback=payload.model_dump(),
                    current_diff=self._bounded_diff(f"{task_key}-a{attempt_id}-repair.diff"),
                )
            # AGENT_FAILURE: feedback already set above

            # ---- retry budget ----------------------------------------------
            failures = self.tasks.consecutive_failures(task_id)
            if failures < self.config.limits.max_attempts:
                continue  # fresh Developer attempt with Attempt Context

            diagnosis_rounds += 1
            outcome = await self._diagnose(
                project_id, task_id, task_key, attempt_id,
                project_ctx, task_ctx, feedback, diagnosis_rounds,
            )
            if outcome is not None:
                return outcome
            # RETRY: clean slate, re-analyze
            feedback_diag = feedback.diagnosis or {}
            feedback = AttemptContext(diagnosis=feedback_diag)
            need_analysis = True

    # ------------------------------------------------------------------

    async def _run_attempt(
        self,
        project_id: int,
        task_id: int,
        task_key: str,
        attempt_id: int,
        project_ctx: ProjectContext,
        task_ctx: TaskContext,
        feedback: AttemptContext,
    ) -> tuple[str, object]:
        # Developer (fresh session, even on repair)
        self.tasks.set_status(task_id, TaskState.EXECUTING, force=True)
        impl = await self.invoker.invoke(
            developer.SPEC,
            project_id,
            project_ctx,
            task_ctx=task_ctx,
            attempt_ctx=feedback,
            task_id=task_id,
            attempt_id=attempt_id,
            artifact_path=self.artifacts.task_artifact_path(task_key, "implementation.json"),
        )
        task_ctx.implementation = impl.model_dump()

        # Test Engineer (fresh session, test files only)
        self.tasks.set_status(task_id, TaskState.TESTING)
        report = await self.invoker.invoke(
            tester.SPEC,
            project_id,
            project_ctx,
            task_ctx=task_ctx,
            task_id=task_id,
            attempt_id=attempt_id,
            artifact_path=self.artifacts.task_artifact_path(task_key, "test-result.json"),
        )
        task_ctx.test_report = report.model_dump()

        # Deterministic verification: exit codes, not agent claims. The
        # command execution is journaled as an operation so a crash mid-run
        # is visible as an unfinished VERIFICATION_COMMAND intent.
        self.tasks.set_status(task_id, TaskState.VERIFYING)
        self.events.emit("TEST_STARTED", project_id=project_id, task_id=task_id, attempt_id=attempt_id)
        verify_op_id = None
        if self.operations:
            verify_op_id = self.operations.record_intent(
                OperationType.VERIFICATION_COMMAND,
                {"commands": self.verifier.commands(), "label": f"{task_key}-a{attempt_id}",
                 # Tree state before the commands run: recovery may only
                 # reset a diff whose hash it has durably recorded.
                 "base_diff_sha256": self._diff_hash()},
                project_id=project_id, task_id=task_id, attempt_id=attempt_id,
            )
        verification = self.verifier.run(
            label=f"{task_key}-a{attempt_id}",
            # After each command the intent's known tree state is refreshed,
            # so a crash mid-verification stays recoverable at command
            # granularity even when commands mutate the tree.
            on_step=(lambda: self.operations.annotate(
                verify_op_id, {"base_diff_sha256": self._diff_hash()}))
            if verify_op_id else None,
        )
        if self.operations and verify_op_id:
            self.operations.record_result(
                verify_op_id, OperationStatus.COMPLETED,
                {"passed": verification.passed,
                 "exit_codes": [s.exit_code for s in verification.steps]},
            )
        self.artifacts.save_model(
            self.artifacts.task_artifact_path(task_key, "verification-result.json"), verification
        )
        # Verification evidence is recorded per attempt so "COMPLETED implies
        # a deterministic PASS" is checkable later (invariant checker).
        self.runs.record_evaluation(
            attempt_id, VERIFY_PASS if verification.passed else VERIFY_FAIL,
            verification.model_dump(),
        )
        if decide_after_verification(verification.passed) == Role.DEVELOPER:
            self.events.emit(
                "TEST_FAILED", project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                payload={"steps": [s.command for s in verification.steps if s.exit_code != 0]},
            )
            return "VERIFY_FAIL", verification

        # Independent Reviewer (fresh session, read-only)
        self.tasks.set_status(task_id, TaskState.REVIEWING)
        self.events.emit("REVIEW_STARTED", project_id=project_id, task_id=task_id, attempt_id=attempt_id)
        diff_preview = self._bounded_diff(f"{task_key}-a{attempt_id}-review.diff",
                                          limit=REVIEW_DIFF_LIMIT)
        extra = (
            "## Change under review (uncommitted diff)\n```diff\n"
            + diff_preview
            + "\n```\n\n## Deterministic verification result\n```json\n"
            + json.dumps(verification.model_dump(), indent=2)[:4000]
            + "\n```"
        )
        review: Review = await self.invoker.invoke(
            reviewer.SPEC,
            project_id,
            project_ctx,
            task_ctx=task_ctx,
            extra=extra,
            task_id=task_id,
            attempt_id=attempt_id,
            artifact_path=self.artifacts.task_artifact_path(task_key, "review.json"),
        )
        self.runs.record_evaluation(attempt_id, review.verdict, review.model_dump(), review.score)

        verdict = ReviewVerdict(review.verdict)
        next_role = decide_after_review(verdict)
        if next_role is None:
            return "PASS", review
        if next_role == Role.PLANNER:
            return "REPLAN", review
        self.events.emit("REVIEW_REPAIR", project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                         payload={"issues": [i.issue for i in review.blocking_issues]})
        return "REPAIR", review

    async def _diagnose(
        self,
        project_id: int,
        task_id: int,
        task_key: str,
        attempt_id: int,
        project_ctx: ProjectContext,
        task_ctx: TaskContext,
        feedback: AttemptContext,
        diagnosis_rounds: int,
    ) -> TaskOutcome | None:
        """Returns an outcome, or None to signal RETRY."""
        self.tasks.set_status(task_id, TaskState.DIAGNOSING, force=True)
        self.events.emit("DIAGNOSIS_STARTED", project_id=project_id, task_id=task_id,
                         payload={"round": diagnosis_rounds})

        if diagnosis_rounds > MAX_DIAGNOSIS_ROUNDS:
            self.checkpoint.discard_working_tree()
            self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
            self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                             payload={"reason": "max diagnosis rounds exceeded"})
            return TaskOutcome.BLOCKED

        attempt_no = self.tasks.get(task_id)["attempt_count"]
        # The failed attempt's diff is already captured in the Attempt Context
        # (and archived here), so the working tree is reset BEFORE diagnosis:
        # a crash mid-diagnosis then leaves a clean tree that startup recovery
        # can handle, instead of a dirty tree with no RUNNING attempt.
        self._archive_and_discard(task_key, f"attempt{attempt_no}")
        try:
            # attempt_id ties the diagnostician's runs to the task so they
            # count toward max_agent_runs_per_task like every other run.
            diagnosis: Diagnosis = await self.invoker.invoke(
                diagnostician.SPEC,
                project_id,
                project_ctx,
                task_ctx=task_ctx,
                attempt_ctx=feedback,
                task_id=task_id,
                attempt_id=attempt_id,
                artifact_path=self.artifacts.diagnostics_path(task_key, attempt_no),
            )
        except (AgentRunFailed, BudgetExceeded) as exc:
            self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
            self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                             payload={"reason": f"diagnosis failed: {exc}"})
            if isinstance(exc, BudgetExceeded) and exc.scope == "project":
                raise
            return TaskOutcome.BLOCKED

        verdict = DiagnosisVerdict(diagnosis.recommendation)
        self.events.emit("DIAGNOSIS_COMPLETED", project_id=project_id, task_id=task_id,
                         payload={"recommendation": verdict.value})

        if verdict == DiagnosisVerdict.RETRY:
            self.tasks.set_status(task_id, TaskState.READY, force=True)
            feedback.diagnosis = diagnosis.model_dump()
            return None
        if verdict == DiagnosisVerdict.SPLIT:
            inserted = self._insert_split_tasks(project_id, task_id, diagnosis)
            if not inserted:
                # A SPLIT with no usable replacement tasks would silently drop
                # the work (SKIPPED is terminal) — treat it as BLOCKED instead.
                self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
                self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                                 payload={"reason": "SPLIT diagnosis contained no replacement tasks"})
                return TaskOutcome.BLOCKED
            self.tasks.set_status(task_id, TaskState.SKIPPED, force=True)
            self.events.emit("TASK_SPLIT", project_id=project_id, task_id=task_id,
                             payload={"new_tasks": inserted})
            return TaskOutcome.SPLIT
        if verdict == DiagnosisVerdict.REPLAN:
            self.tasks.set_status(task_id, TaskState.PENDING, force=True)
            return TaskOutcome.REPLAN
        self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
        self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                         payload={"reason": "diagnostician verdict BLOCKED"})
        return TaskOutcome.BLOCKED

    # ------------------------------------------------------------------

    def _complete_task(
        self, project_id: int, task_id: int, task_key: str, attempt_id: int, task_row: sqlite3.Row
    ) -> TaskOutcome:
        # Completion evidence is deterministic, not an agent claim: the
        # attempt must carry a recorded verification PASS and a Reviewer
        # PASS before the checkpoint commit may happen.
        if not self.runs.has_evaluation(attempt_id, VERIFY_PASS):
            raise RuntimeError(
                f"refusing to complete {task_key}: attempt {attempt_id} has no "
                "recorded deterministic verification PASS"
            )
        if not self.runs.has_evaluation(attempt_id, ReviewVerdict.PASS.value):
            raise RuntimeError(
                f"refusing to complete {task_key}: attempt {attempt_id} has no "
                "recorded Reviewer PASS"
            )
        commit_hash = self.checkpoint.commit_task(
            task_key, task_row["title"],
            project_id=project_id, task_id=task_id, attempt_id=attempt_id,
        )
        # The GIT_COMMIT result and every task-completion update land in ONE
        # transaction: either the operation stays PENDING (and recovery
        # reconciles the whole completion from the commit trailer) or all of
        # it is durable. No window where the journal says done but the task
        # state was lost.
        commit_op_id = self.checkpoint.pending_operation_id
        with self.tasks.db.transaction():
            if commit_op_id and self.operations:
                self.operations.record_result(
                    commit_op_id, OperationStatus.COMPLETED, {"commit": commit_hash}
                )
            self.tasks.finish_attempt(attempt_id, AttemptState.PASSED)
            if commit_hash:
                self.tasks.set_commit(task_id, commit_hash)
            self.tasks.set_status(task_id, TaskState.COMPLETED, force=True)
            self.events.emit("TASK_COMPLETED", project_id=project_id, task_id=task_id,
                             attempt_id=attempt_id, operation_id=commit_op_id,
                             payload={"commit": commit_hash})
        logger.info("task %s completed (commit %s)", task_key, commit_hash)
        return TaskOutcome.COMPLETED

    def _insert_split_tasks(
        self, project_id: int, task_id: int, diagnosis: Diagnosis
    ) -> list[str]:
        """Insert the diagnosis's replacement tasks; returns the inserted keys."""
        original = self.tasks.get(task_id)
        base_seq = original["sequence"]
        all_tasks = self.tasks.list_for_project(project_id)
        next_seqs = sorted(t["sequence"] for t in all_tasks if t["sequence"] > base_seq)
        upper = next_seqs[0] if next_seqs else base_seq + 100
        count = len(diagnosis.split_tasks)
        inserted: list[str] = []
        for index, planned in enumerate(diagnosis.split_tasks, start=1):
            seq = base_seq + max(1, (upper - base_seq) * index // (count + 1))
            if self.tasks.get_by_key(project_id, planned.task_key) is not None:
                continue
            self.tasks.create(
                project_id,
                planned.task_key,
                planned.title,
                planned.goal,
                planned.acceptance_criteria,
                planned.dependencies,
                sequence=seq,
            )
            inserted.append(planned.task_key)
        return inserted

    def _archive_and_discard(self, task_key: str, label: str) -> None:
        """Save the current dirty diff as an artifact, then reset the tree.

        Archived as a byte-exact, re-applicable patch — the archive is the
        only copy once the tree is reset, so a failed archive aborts the
        reset instead of proceeding without one.
        """
        diff = self.git.snapshot_dirty_bytes()
        if diff.strip():
            self.artifacts.save_bytes(
                self.artifacts.root / "diagnostics" / f"{task_key}-{label}.diff", diff
            )
        self.checkpoint.discard_working_tree()

    def _finish_running_attempts(
        self, task_id: int, status: AttemptState = AttemptState.INTERRUPTED
    ) -> None:
        for row in self.tasks.db.query_all(
            "SELECT id FROM task_attempts WHERE task_id = ? AND status = 'RUNNING'", (task_id,)
        ):
            self.tasks.finish_attempt(row["id"], status)

    def _safe_diff(self) -> str:
        try:
            return self.git.full_dirty_diff()
        except Exception as exc:
            logger.warning("could not capture diff: %s", exc)
            return ""

    def _diff_hash(self) -> str:
        import hashlib

        try:
            diff = self.git.dirty_diff_readonly()
        except Exception as exc:
            logger.warning("could not hash diff: %s", exc)
            return ""
        return hashlib.sha256(diff.encode("utf-8")).hexdigest()

    def _bounded_diff(self, spill_name: str, limit: int | None = None) -> str:
        """The current dirty diff, spilled to an artifact when oversized.

        The full diff is always retained on disk; agent context receives at
        most a bounded head/tail preview plus the artifact locator.
        """
        diff = self._safe_diff()
        threshold = limit or self.config.limits.max_inline_output_chars
        if len(diff) <= threshold:
            return diff
        spilled = self.artifacts.spill_text_output(spill_name, diff, threshold=threshold)
        return spilled.render()

    def _build_task_ctx(self, task_row: sqlite3.Row) -> TaskContext:
        return TaskContext(
            task_key=task_row["task_key"],
            title=task_row["title"],
            goal=task_row["goal"],
            acceptance_criteria=json.loads(task_row["acceptance_criteria"] or "[]"),
            dependencies=json.loads(task_row["dependencies"] or "[]"),
        )
