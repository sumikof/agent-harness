"""Task Loop: Analyst -> Developer -> Tester -> Verification -> Reviewer
-> serialized Integration.

All routing decisions here are deterministic Python. Agents think;
the harness decides who runs next.

Parallel-execution model: every task attempt cycle gets an ISOLATED git
worktree + branch (created from the current integration HEAD). All roles
of the cycle share that worktree's filesystem state — conversations stay
fresh, uncommitted diffs are visible to Tester/Reviewer. After Reviewer
PASS the harness commits on the task branch (task_commit) and the
IntegrationManager merges it — strictly serialized — into the
integration branch (integration_commit = the official checkpoint).
Integration conflicts are a normal outcome and route into a fresh repair
attempt based on the NEW integration HEAD.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from enum import StrEnum

from ..agents import analyst, developer, diagnostician, reviewer, tester
from ..artifacts.manager import ArtifactManager
from ..artifacts.schemas import Diagnosis, Review, VerificationResult
from ..config import HarnessConfig
from ..context.attempt_context import AttemptContext
from ..context.project_context import ProjectContext
from ..context.task_context import TaskContext
from ..database.event_repository import EventRepository, EventType
from ..database.operation_repository import (
    OperationRepository,
    OperationStatus,
    OperationType,
)
from ..database.run_repository import VERIFY_FAIL, VERIFY_PASS, RunRepository
from ..database.task_repository import TaskRepository
from ..git.checkpoint import CheckpointManager
from ..git.integration import IntegrationManager, IntegrationStatus
from ..git.repository import GitRepository
from ..git.worktree import WorktreeHandle, WorktreeManager
from ..verification.runner import VerificationRunner
from .agent_invoker import AgentConfigurationError, AgentInvoker, AgentRunFailed
from .budget import BudgetExceeded
from .resources import ResourcePools
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
CONFLICT_DIFF_LIMIT = 20000


class TaskOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    REPLAN = "REPLAN"
    SPLIT = "SPLIT"
    BLOCKED = "BLOCKED"


@dataclass
class TaskEnv:
    """Per-cycle execution environment: one worktree, one verifier."""

    handle: WorktreeHandle
    checkpoint: CheckpointManager
    verifier: VerificationRunner

    @property
    def git(self) -> GitRepository:
        return self.handle.repo


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
        worktrees: WorktreeManager,
        integration: IntegrationManager,
        pools: ResourcePools,
        operations: OperationRepository | None = None,
    ):
        self.config = config
        self.invoker = invoker
        self.tasks = tasks
        self.runs = runs
        self.events = events
        self.artifacts = artifacts
        self.git = git                     # main checkout = integration branch
        self.worktrees = worktrees
        self.integration = integration
        self.pools = pools
        self.operations = operations

    # ------------------------------------------------------------------

    def _create_env(self, task_key: str, cycle: int) -> TaskEnv:
        base_commit = self.git.head_commit()
        if base_commit is None:
            raise RuntimeError("cannot create a task worktree without a baseline commit")
        handle = self.worktrees.create(task_key, cycle, base_commit)
        verifier = VerificationRunner(
            self.config.verification,
            handle.path,
            self.config.logs_path / "verification" / f"{task_key}-c{cycle}",
        )
        return TaskEnv(
            handle=handle,
            checkpoint=CheckpointManager(handle.repo, self.operations),
            verifier=verifier,
        )

    def _dispose_env(self, env: TaskEnv | None) -> None:
        if env is not None:
            self.worktrees.remove(env.handle)

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
        integration_repairs = 0
        need_analysis = True
        env: TaskEnv | None = None

        while True:
            try:
                if env is None:
                    cycle = (self.tasks.get(task_id)["attempt_count"] or 0) + 1
                    env = self._create_env(task_key, cycle)
                # The attempt is opened BEFORE analysis so that an Analyst
                # failure is recorded as a failed attempt and counts toward
                # the retry limit — otherwise a failing analysis would loop
                # outside every budget except wall clock.
                attempt_id = self.tasks.start_attempt(
                    task_id, env.git.head_commit(),
                    worktree_path=str(env.handle.path), branch=env.handle.branch,
                )
                feedback.attempt_no = self.tasks.get(task_id)["attempt_count"]

                if need_analysis:
                    self.tasks.set_status(task_id, TaskState.ANALYZING, force=True)
                    brief = await self._invoke(
                        analyst.SPEC, project_id, project_ctx, env,
                        task_ctx=task_ctx, attempt_ctx=feedback,
                        task_id=task_id, attempt_id=attempt_id,
                        artifact_path=self.artifacts.task_artifact_path(task_key, "task-brief.json"),
                    )
                    task_ctx.task_brief = brief.model_dump()
                    task_ctx.relevant_files = list(brief.files)
                    need_analysis = False

                kind, payload = await self._run_attempt(
                    project_id, task_id, task_key, attempt_id, project_ctx, task_ctx,
                    feedback, env,
                )
            except BudgetExceeded as exc:
                # A pause must leave no dangling worktree state. The diff is
                # archived first so nothing is lost; the worktree is removed
                # (other tasks' worktrees are untouched).
                self._archive_and_dispose(env, task_key, "budget-paused")
                env = None
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
                self._archive_and_dispose(env, task_key, "config-error")
                env = None
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
                    current_diff=self._bounded_diff(env, f"{task_key}-a{attempt_id}-aborted.diff"),
                )
                kind, payload = ("AGENT_FAILURE", None)

            # ---- deterministic routing --------------------------------------
            if kind == "PASS":
                outcome, conflict_feedback = await self._complete_and_integrate(
                    project_id, task_id, task_key, attempt_id, task_row, env, payload
                )
                if outcome is not None:
                    self._dispose_env(env)
                    env = None
                    return outcome
                # Integration conflict: fresh repair attempt on the NEW
                # integration HEAD. The old worktree is already archived
                # and removed by _complete_and_integrate.
                env = None
                integration_repairs += 1
                if integration_repairs > self.config.limits.max_integration_repairs:
                    self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
                    self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                                     payload={"reason": "max integration repairs exceeded"})
                    return TaskOutcome.BLOCKED
                feedback = conflict_feedback
                continue

            if kind == "REPLAN":
                self.tasks.finish_attempt(attempt_id, AttemptState.FAILED)
                self._archive_and_dispose(env, task_key, f"a{attempt_id}-replan")
                env = None
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
                    current_diff=self._bounded_diff(env, f"{task_key}-a{attempt_id}-verify-fail.diff"),
                )
            elif kind == "REPAIR":
                self.tasks.finish_attempt(attempt_id, AttemptState.FAILED)
                self.tasks.set_status(task_id, TaskState.REPAIR_REQUIRED, force=True)
                feedback = AttemptContext(
                    review_feedback=payload.model_dump(),
                    current_diff=self._bounded_diff(env, f"{task_key}-a{attempt_id}-repair.diff"),
                )
            # AGENT_FAILURE: feedback already set above

            # ---- retry budget ----------------------------------------------
            failures = self.tasks.consecutive_failures(task_id)
            if failures < self.config.limits.max_attempts:
                # Fresh Developer attempt with Attempt Context; the SAME
                # worktree is kept — the repair continues on the dirty state
                # the previous roles produced.
                continue

            diagnosis_rounds += 1
            outcome = await self._diagnose(
                project_id, task_id, task_key, attempt_id,
                project_ctx, task_ctx, feedback, diagnosis_rounds, env,
            )
            env = None  # diagnosis always archives + disposes the worktree
            if outcome is not None:
                return outcome
            # RETRY: clean slate, re-analyze in a fresh worktree
            feedback_diag = feedback.diagnosis or {}
            feedback = AttemptContext(diagnosis=feedback_diag)
            need_analysis = True

    # ------------------------------------------------------------------

    async def _invoke(
        self, spec, project_id, project_ctx, env: TaskEnv, **kwargs
    ):
        """invoker.invoke bound to this task's worktree."""
        attempt_no = None
        if kwargs.get("attempt_id") is not None:
            row = self.tasks.get_attempt(kwargs["attempt_id"])
            attempt_no = row["attempt_no"] if row else None
        volatile = {
            "task_branch": env.handle.branch,
            "attempt_no": attempt_no,
        }
        return await self.invoker.invoke(
            spec, project_id, project_ctx,
            workdir=env.handle.path, git=env.git, volatile=volatile, **kwargs,
        )

    async def _run_attempt(
        self,
        project_id: int,
        task_id: int,
        task_key: str,
        attempt_id: int,
        project_ctx: ProjectContext,
        task_ctx: TaskContext,
        feedback: AttemptContext,
        env: TaskEnv,
    ) -> tuple[str, object]:
        # Developer (fresh session, even on repair)
        self.tasks.set_status(task_id, TaskState.EXECUTING, force=True)
        impl = await self._invoke(
            developer.SPEC, project_id, project_ctx, env,
            task_ctx=task_ctx, attempt_ctx=feedback,
            task_id=task_id, attempt_id=attempt_id,
            artifact_path=self.artifacts.task_artifact_path(task_key, "implementation.json"),
        )
        task_ctx.implementation = impl.model_dump()

        # Test Engineer (fresh session, test files only)
        self.tasks.set_status(task_id, TaskState.TESTING)
        report = await self._invoke(
            tester.SPEC, project_id, project_ctx, env,
            task_ctx=task_ctx,
            task_id=task_id, attempt_id=attempt_id,
            artifact_path=self.artifacts.task_artifact_path(task_key, "test-result.json"),
        )
        task_ctx.test_report = report.model_dump()

        # Deterministic verification: exit codes, not agent claims. The
        # command execution is journaled as an operation so a crash mid-run
        # is visible as an unfinished VERIFICATION_COMMAND intent. It runs
        # in the heavy_test pool on a worker thread: builds/tests consume
        # host CPU, not GPU inference slots, and other agents keep inferring.
        self.tasks.set_status(task_id, TaskState.VERIFYING)
        self.events.emit("TEST_STARTED", project_id=project_id, task_id=task_id, attempt_id=attempt_id)
        verify_op_id = None
        if self.operations:
            verify_op_id = self.operations.record_intent(
                OperationType.VERIFICATION_COMMAND,
                {"commands": env.verifier.commands(), "label": f"{task_key}-a{attempt_id}",
                 "worktree": str(env.handle.path),
                 # Tree state before the commands run: recovery may only
                 # reset a diff whose hash it has durably recorded.
                 "base_diff_sha256": self._diff_hash(env)},
                project_id=project_id, task_id=task_id, attempt_id=attempt_id,
            )
        on_step = (
            (lambda: self.operations.annotate(
                verify_op_id, {"base_diff_sha256": self._diff_hash(env)}))
            if verify_op_id else None
        )
        async with self.pools.heavy_test:
            verification: VerificationResult = await asyncio.to_thread(
                env.verifier.run, f"{task_key}-a{attempt_id}", on_step
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

        # Independent Reviewer (fresh session, read-only, NO developer
        # conversation — only artifacts, diff and verification evidence)
        self.tasks.set_status(task_id, TaskState.REVIEWING)
        self.events.emit("REVIEW_STARTED", project_id=project_id, task_id=task_id, attempt_id=attempt_id)
        diff_preview = self._bounded_diff(env, f"{task_key}-a{attempt_id}-review.diff",
                                          limit=REVIEW_DIFF_LIMIT)
        extra = (
            "## Change under review (uncommitted diff)\n```diff\n"
            + diff_preview
            + "\n```\n\n## Deterministic verification result\n```json\n"
            + json.dumps(verification.model_dump(), indent=2)[:4000]
            + "\n```"
        )
        review: Review = await self._invoke(
            reviewer.SPEC, project_id, project_ctx, env,
            task_ctx=task_ctx, extra=extra,
            task_id=task_id, attempt_id=attempt_id,
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
        env: TaskEnv | None,
    ) -> TaskOutcome | None:
        """Returns an outcome, or None to signal RETRY."""
        self.tasks.set_status(task_id, TaskState.DIAGNOSING, force=True)
        self.events.emit("DIAGNOSIS_STARTED", project_id=project_id, task_id=task_id,
                         payload={"round": diagnosis_rounds})

        if diagnosis_rounds > MAX_DIAGNOSIS_ROUNDS:
            self._archive_and_dispose(env, task_key, "max-diagnosis")
            self.tasks.set_status(task_id, TaskState.BLOCKED, force=True)
            self.events.emit("TASK_BLOCKED", project_id=project_id, task_id=task_id,
                             payload={"reason": "max diagnosis rounds exceeded"})
            return TaskOutcome.BLOCKED

        attempt_no = self.tasks.get(task_id)["attempt_count"]
        # The failed attempt's diff is already captured in the Attempt Context
        # (and archived here); the worktree is disposed BEFORE diagnosis so a
        # crash mid-diagnosis leaves no dangling worktree. The Diagnostician
        # runs read-only against the MAIN checkout.
        self._archive_and_dispose(env, task_key, f"attempt{attempt_no}")
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

    async def _complete_and_integrate(
        self,
        project_id: int,
        task_id: int,
        task_key: str,
        attempt_id: int,
        task_row: sqlite3.Row,
        env: TaskEnv,
        review: Review,
    ) -> tuple[TaskOutcome | None, AttemptContext | None]:
        """Reviewer PASSed: commit on the task branch, then integrate.

        Returns (outcome, None) when the task settled, or
        (None, conflict_feedback) to signal a fresh repair attempt.
        """
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
        # 1. task_commit on the isolated task branch (journaled GIT_COMMIT).
        task_commit = env.checkpoint.commit_task(
            task_key, task_row["title"],
            project_id=project_id, task_id=task_id, attempt_id=attempt_id,
        )
        commit_op_id = env.checkpoint.pending_operation_id
        with self.tasks.db.transaction():
            if commit_op_id and self.operations:
                self.operations.record_result(
                    commit_op_id, OperationStatus.COMPLETED, {"commit": task_commit}
                )
            self.tasks.finish_attempt(attempt_id, AttemptState.PASSED)
            if task_commit:
                self.tasks.set_task_commit(task_id, task_commit)

        if task_commit is None:
            # A task may legitimately produce no diff (verification-only):
            # nothing to integrate, current integration HEAD is the checkpoint.
            with self.tasks.db.transaction():
                self.tasks.set_status(task_id, TaskState.COMPLETED, force=True)
                self.events.emit("TASK_COMPLETED", project_id=project_id, task_id=task_id,
                                 attempt_id=attempt_id, payload={"commit": None})
            return TaskOutcome.COMPLETED, None

        # 2. Serialized integration into the integration branch.
        self.tasks.set_status(task_id, TaskState.INTEGRATING, force=True)
        self.events.emit(EventType.INTEGRATION_STARTED, project_id=project_id,
                         task_id=task_id, attempt_id=attempt_id,
                         payload={"task_commit": task_commit})
        async with self.pools.git_integration:
            outcome = await self.integration.integrate(
                task_key=task_key, task_commit=task_commit, title=task_row["title"],
                project_id=project_id, task_id=task_id, attempt_id=attempt_id,
            )

        if outcome.status in (IntegrationStatus.MERGED, IntegrationStatus.NOOP):
            # Integration result + completion state land in ONE transaction
            # (crash between merge and here is reconciled by trailer).
            integration_op_id = self.integration.pending_operation_id
            with self.tasks.db.transaction():
                if integration_op_id and self.operations:
                    self.operations.record_result(
                        integration_op_id, OperationStatus.COMPLETED,
                        {"integration_commit": outcome.integration_commit},
                    )
                if outcome.integration_commit:
                    self.tasks.set_integration_commit(task_id, outcome.integration_commit)
                self.tasks.set_status(task_id, TaskState.COMPLETED, force=True)
                self.events.emit(
                    EventType.INTEGRATION_COMPLETED, project_id=project_id,
                    task_id=task_id, attempt_id=attempt_id,
                    operation_id=integration_op_id,
                    payload={"task_commit": task_commit,
                             "integration_commit": outcome.integration_commit},
                )
                self.events.emit("TASK_COMPLETED", project_id=project_id, task_id=task_id,
                                 attempt_id=attempt_id,
                                 payload={"commit": outcome.integration_commit})
            logger.info("task %s completed (integration commit %s)",
                        task_key, outcome.integration_commit)
            return TaskOutcome.COMPLETED, None

        # 3. Conflict: normal path. Archive the original change, dispose the
        #    worktree, hand a fresh repair attempt the conflict context.
        original_diff = self._commit_diff(env, env.handle.base_commit, task_commit)
        archive = self.artifacts.root / "diagnostics" / f"{task_key}-integration-conflict.diff"
        if original_diff.strip():
            self.artifacts.save_text(archive, original_diff)
        self.tasks.set_status(task_id, TaskState.INTEGRATION_CONFLICT, force=True)
        self.events.emit(
            EventType.INTEGRATION_CONFLICT, project_id=project_id, task_id=task_id,
            attempt_id=attempt_id,
            payload={"task_commit": task_commit, "files": outcome.conflict_files},
        )
        self._dispose_env(env)
        feedback = AttemptContext(
            integration_conflict={
                "conflict_files": outcome.conflict_files,
                "original_base_commit": env.handle.base_commit,
                "original_task_commit": task_commit,
                "original_diff": original_diff[:CONFLICT_DIFF_LIMIT],
                "archived_diff_artifact": self.artifacts.relpath(archive),
            },
        )
        return None, feedback

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

    # ------------------------------------------------------------------

    def _archive_and_dispose(self, env: TaskEnv | None, task_key: str, label: str) -> None:
        """Save the worktree's dirty diff as an artifact, then remove the
        worktree entirely. Only THIS task's worktree is touched.

        Archived as a byte-exact, re-applicable patch — the archive is the
        only copy once the worktree is removed, so a failed archive aborts
        the removal instead of proceeding without one.
        """
        if env is None:
            return
        diff = env.git.snapshot_dirty_bytes()
        archive_path = self.artifacts.root / "diagnostics" / f"{task_key}-{label}.diff"
        if diff.strip():
            self.artifacts.save_bytes(archive_path, diff)
        # Real-file tar regardless of the patch: a clean filter can render
        # the diff empty while the on-disk bytes still differ.
        self.artifacts.archive_worktree_files(
            archive_path.with_suffix(".files.tar"), env.git.path,
            env.git.changed_paths(),
        )
        self._dispose_env(env)

    def _finish_running_attempts(
        self, task_id: int, status: AttemptState = AttemptState.INTERRUPTED
    ) -> None:
        for row in self.tasks.db.query_all(
            "SELECT id FROM task_attempts WHERE task_id = ? AND status = 'RUNNING'", (task_id,)
        ):
            self.tasks.finish_attempt(row["id"], status)

    def _safe_diff(self, env: TaskEnv | None) -> str:
        if env is None:
            return ""
        try:
            return env.git.full_dirty_diff()
        except Exception as exc:
            logger.warning("could not capture diff: %s", exc)
            return ""

    def _diff_hash(self, env: TaskEnv) -> str:
        try:
            return env.git.dirty_state_hash()
        except Exception as exc:
            logger.warning("could not hash dirty state: %s", exc)
            return ""

    def _commit_diff(self, env: TaskEnv, base: str, head: str) -> str:
        try:
            return env.git._run("diff", "--no-color", f"{base}..{head}").stdout
        except Exception as exc:
            logger.warning("could not capture commit diff: %s", exc)
            return ""

    def _bounded_diff(self, env: TaskEnv | None, spill_name: str, limit: int | None = None) -> str:
        """The current dirty diff, spilled to an artifact when oversized.

        The full diff is always retained on disk; agent context receives at
        most a bounded head/tail preview plus the artifact locator.
        """
        diff = self._safe_diff(env)
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
