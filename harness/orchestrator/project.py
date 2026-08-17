"""Project Loop: plan once, then run tasks strictly one at a time.

Owns wiring of every component and the project-level state machine.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ..agents import planner
from ..artifacts.manager import ArtifactManager
from ..artifacts.schemas import ProjectPlan
from ..config import HarnessConfig
from ..context.builder import ContextBuilder, render_plan_for_replan
from ..context.project_context import DEFAULT_FORBIDDEN_OPERATIONS, ProjectContext
from ..database.connection import Database
from ..database.event_repository import EventRepository, EventType
from ..database.operation_repository import OperationRepository
from ..database.project_repository import ProjectRepository
from ..database.run_repository import RunRepository
from ..database.task_repository import TaskRepository
from ..git.checkpoint import CheckpointManager
from ..git.repository import GitRepository
from ..verification.runner import VerificationRunner
from ..workspace_lock import WorkspaceLock
from .agent_invoker import AgentConfigurationError, AgentInvoker, AgentRunFailed
from .budget import BudgetExceeded, BudgetManager
from .invariants import InvariantChecker, Severity
from .recovery import RecoveryIntegrityError, RecoveryManager
from .state_machine import ProjectState, TaskState
from .task_runner import TaskOutcome, TaskRunner

logger = logging.getLogger(__name__)

MAX_REPLANS = 3


class ProjectOrchestrator:
    def __init__(self, config: HarnessConfig):
        self.config = config
        config.workspace_path.mkdir(parents=True, exist_ok=True)
        config.logs_path.mkdir(parents=True, exist_ok=True)

        # One process per workspace: without this, a second process's
        # startup recovery would treat the first process's LIVE RUNNING run
        # as a crash leftover — interrupting it, freeing the max-1-agent
        # slot, and resetting a tree an agent is still working in.
        self.lock = WorkspaceLock(config.workspace_path / "harness.lock")
        self.lock.acquire()

        self.db = Database(config.db_path)
        self.events = EventRepository(self.db)
        # Repositories share the EventRepository so every state change and
        # its ledger event are written in one transaction.
        self.projects = ProjectRepository(self.db, self.events)
        self.tasks = TaskRepository(self.db, self.events)
        self.runs = RunRepository(self.db)
        self.operations = OperationRepository(self.db, self.events)
        self.artifacts = ArtifactManager(config.artifacts_path)
        self.git = GitRepository(config.repository_path)
        self.checkpoint = CheckpointManager(self.git, self.operations)
        self.verifier = VerificationRunner(
            config.verification, config.repository_path, config.logs_path / "verification"
        )
        self.context_builder = ContextBuilder(config.prompts_path)
        self.budget = BudgetManager(config, self.projects, self.tasks, self.runs)
        self.invoker = AgentInvoker(
            config, self.context_builder, self.artifacts, self.runs, self.events,
            self.budget, self.operations, self.git,
        )
        self.task_runner = TaskRunner(
            config, self.invoker, self.tasks, self.runs, self.events,
            self.artifacts, self.git, self.checkpoint, self.verifier, self.operations,
        )
        self.recovery = RecoveryManager(
            self.tasks, self.events, self.artifacts, self.git, self.checkpoint,
            operations=self.operations, runs=self.runs,
        )
        self.invariants = InvariantChecker(self.db, self.git, config.artifacts_path)
        self._replan_for_new_goal = False

    # ------------------------------------------------------------------

    def ensure_project(self) -> sqlite3.Row:
        # Validate the working copy BEFORE touching persistent state, so a
        # mistyped path never binds a project row to an invalid identity.
        if not self.git.is_repo():
            raise RuntimeError(
                f"{self.config.repository_path} is not the root of a git repository. "
                "Place or clone the target repository there first (a subdirectory of "
                "another checkout is not accepted)."
            )
        branch = self.git.current_branch()
        if branch != self.config.project.base_branch:
            raise RuntimeError(
                f"repository is on branch '{branch}' but base_branch is "
                f"'{self.config.project.base_branch}'. Check out the base branch first — "
                "harness checkpoints are committed to the currently checked-out branch."
            )

        self._replan_for_new_goal = False
        row = self.projects.get_by_name(self.config.project.name)
        if row is None:
            project_id = self.projects.create(
                self.config.project.name,
                str(self.config.repository_path),
                self.config.project.base_branch,
                self.config.project.goal,
                self.config.budget.project_usd,
            )
            self.events.emit("PROJECT_CREATED", project_id=project_id,
                             payload={"name": self.config.project.name})
            row = self.projects.get(project_id)
        else:
            row = self._validate_resumed_project(row)
        if self.git.head_commit() is None:
            # Checkpoint/recovery semantics need a HEAD to reset to. Commit
            # whatever the repository starts with as the baseline; ignored
            # files stay untouched on disk.
            self.git.add_all()
            baseline = self.git.commit("harness: baseline commit", allow_empty=True)
            self.events.emit("BASELINE_COMMITTED", project_id=row["id"],
                             payload={"commit": baseline})
            logger.info("created baseline commit %s in commitless repository", baseline[:8])
        return row

    def _validate_resumed_project(self, row: sqlite3.Row) -> sqlite3.Row:
        """Check persisted project identity against the current configuration."""
        project_id = row["id"]
        stored_repo = Path(row["repository"]).resolve()
        configured_repo = self.config.repository_path.resolve()
        identity_changed = (
            stored_repo != configured_repo
            or row["base_branch"] != self.config.project.base_branch
        )
        if identity_changed:
            untouched = (
                row["status"] == ProjectState.CREATED.value
                and not self.tasks.list_for_project(project_id)
            )
            if not untouched:
                raise RuntimeError(
                    f"project '{self.config.project.name}' in this workspace was created for "
                    f"repository={row['repository']} (base_branch={row['base_branch']}), but the "
                    f"config now points at {configured_repo} (base_branch="
                    f"{self.config.project.base_branch}). Use a new project name or a fresh "
                    "workspace instead of reusing stale state."
                )
            # Nothing has run yet — treat this as correcting a misconfiguration.
            self.projects.update_identity(
                project_id, str(self.config.repository_path), self.config.project.base_branch
            )
            self.events.emit("PROJECT_IDENTITY_CORRECTED", project_id=project_id,
                             payload={"repository": str(self.config.repository_path),
                                      "base_branch": self.config.project.base_branch})
        if row["goal"] != self.config.project.goal:
            if row["status"] in (ProjectState.COMPLETED.value, ProjectState.FAILED.value):
                raise RuntimeError(
                    f"project '{self.config.project.name}' is already {row['status']}; "
                    "a new goal needs a new project name or a fresh workspace."
                )
            # Goal text is content, not identity — adopt it, but the existing
            # plan was made for the old goal, so unfinished work is replanned.
            self.projects.update_goal(project_id, self.config.project.goal)
            self.events.emit("PROJECT_GOAL_UPDATED", project_id=project_id,
                             payload={"goal": self.config.project.goal})
            self._replan_for_new_goal = True
        return self.projects.get(project_id)

    def project_context(self) -> ProjectContext:
        return ProjectContext(
            name=self.config.project.name,
            goal=self.config.project.goal,
            repository_path=str(self.config.repository_path),
            base_branch=self.config.project.base_branch,
            forbidden_operations=list(DEFAULT_FORBIDDEN_OPERATIONS),
        )

    # ------------------------------------------------------------------

    async def run(self) -> ProjectState:
        project = self.ensure_project()
        project_id = project["id"]
        try:
            self.recovery.recover(project)
        except RecoveryIntegrityError as exc:
            # The state plane itself cannot be trusted; never run on top of it.
            logger.error("recovery integrity failure: %s", exc)
            self.projects.set_status(project_id, ProjectState.BLOCKED, force=True)
            self.events.emit(EventType.PROJECT_BLOCKED, project_id=project_id,
                             payload={"reason": str(exc)})
            return ProjectState.BLOCKED

        # Invariant validation after recovery: ERROR-severity violations
        # block loudly instead of being silently repaired; warnings are
        # recorded to the ledger.
        violations = self.invariants.check_all(project_id)
        errors = [v for v in violations if v.severity == Severity.ERROR]
        for violation in violations:
            self.events.emit(
                EventType.INVARIANT_VIOLATION if violation.severity == Severity.ERROR
                else EventType.INVARIANT_WARNING,
                project_id=project_id,
                payload={"code": violation.code, "message": violation.message},
            )
            logger.log(
                logging.ERROR if violation.severity == Severity.ERROR else logging.WARNING,
                "invariant %s: %s", violation.code, violation.message,
            )
        if errors:
            self.projects.set_status(project_id, ProjectState.BLOCKED, force=True)
            return ProjectState.BLOCKED

        project = self.projects.get(project_id)
        state = ProjectState(project["status"])
        project_ctx = self.project_context()

        if state in (ProjectState.COMPLETED, ProjectState.FAILED):
            logger.info("project already %s", state)
            return state

        try:
            if state == ProjectState.CREATED or (
                state == ProjectState.PLANNING and not self.tasks.list_for_project(project_id)
            ):
                if state == ProjectState.CREATED:
                    self.projects.set_status(project_id, ProjectState.PLANNING)
                await self._plan(project_id, project_ctx, replan=False)
                self.projects.set_status(project_id, ProjectState.READY)
                state = ProjectState.READY

            elif self._replan_for_new_goal and self.tasks.list_for_project(project_id):
                # The stored plan was produced for the previous goal — revise
                # it before resuming execution.
                self.projects.set_status(project_id, ProjectState.REPLANNING, force=True)
                await self._plan(project_id, project_ctx, replan=True)
                state = ProjectState.REPLANNING

            if state in (ProjectState.READY, ProjectState.PAUSED, ProjectState.BLOCKED,
                         ProjectState.PLANNING, ProjectState.REPLANNING):
                self.projects.set_status(project_id, ProjectState.RUNNING, force=True)

            return await self._run_task_loop(project_id, project_ctx)
        except BudgetExceeded as exc:
            logger.warning("project paused: %s", exc)
            self.projects.set_status(project_id, ProjectState.PAUSED, force=True)
            self.events.emit("PROJECT_PAUSED", project_id=project_id, payload={"reason": str(exc)})
            return ProjectState.PAUSED
        except AgentConfigurationError as exc:
            # Misconfiguration (missing capability, bad provider) is not
            # retryable at any layer — fail the project explicitly.
            logger.error("project failed (configuration): %s", exc)
            self.projects.set_status(project_id, ProjectState.FAILED, force=True)
            self.events.emit("PROJECT_FAILED", project_id=project_id,
                             payload={"reason": f"configuration error: {exc.detail}"})
            return ProjectState.FAILED
        except AgentRunFailed as exc:
            logger.error("project failed: %s", exc)
            self.projects.set_status(project_id, ProjectState.FAILED, force=True)
            self.events.emit("PROJECT_FAILED", project_id=project_id, payload={"reason": str(exc)})
            return ProjectState.FAILED

    async def _run_task_loop(self, project_id: int, project_ctx: ProjectContext) -> ProjectState:
        replans = 0
        while True:
            self.budget.check_project(project_id)
            task = self.tasks.next_runnable(project_id)
            if task is None:
                return await self._finalize(project_id, project_ctx)

            outcome = await self.task_runner.run_task(project_id, task, project_ctx)
            logger.info("task %s outcome: %s", task["task_key"], outcome)

            if outcome == TaskOutcome.REPLAN:
                replans += 1
                if replans > MAX_REPLANS:
                    self.projects.set_status(project_id, ProjectState.FAILED, force=True)
                    self.events.emit("PROJECT_FAILED", project_id=project_id,
                                     payload={"reason": "max replans exceeded"})
                    return ProjectState.FAILED
                self.projects.set_status(project_id, ProjectState.REPLANNING, force=True)
                await self._plan(project_id, project_ctx, replan=True)
                self.projects.set_status(project_id, ProjectState.RUNNING)
            # COMPLETED / SPLIT / BLOCKED: just take the next runnable task

    async def _finalize(self, project_id: int, project_ctx: ProjectContext) -> ProjectState:
        # Any non-terminal task blocks completion: BLOCKED tasks, but also
        # PENDING/READY tasks that next_runnable() could not schedule (missing
        # dependency or dependency cycle). Finishing "successfully" while work
        # remains would silently drop it.
        terminal = {TaskState.COMPLETED.value, TaskState.SKIPPED.value}
        unfinished = [
            t["task_key"]
            for t in self.tasks.list_for_project(project_id)
            if t["status"] not in terminal
        ]
        if unfinished:
            self.projects.set_status(project_id, ProjectState.BLOCKED, force=True)
            self.events.emit("PROJECT_BLOCKED", project_id=project_id,
                             payload={"unfinished_tasks": unfinished})
            logger.warning("project blocked; unfinished tasks: %s", ", ".join(unfinished))
            return ProjectState.BLOCKED

        self.projects.set_status(project_id, ProjectState.FINAL_VERIFICATION, force=True)
        self.events.emit("FINAL_VERIFICATION_STARTED", project_id=project_id)
        result = self.verifier.run(label="final")
        self.artifacts.save_model(self.artifacts.root / "final-verification.json", result)
        if result.passed:
            self.projects.set_status(project_id, ProjectState.COMPLETED)
            self.events.emit("PROJECT_COMPLETED", project_id=project_id)
            logger.info("project completed")
            return ProjectState.COMPLETED
        self.projects.set_status(project_id, ProjectState.FAILED, force=True)
        self.events.emit("PROJECT_FAILED", project_id=project_id,
                         payload={"reason": "final verification failed"})
        return ProjectState.FAILED

    # ------------------------------------------------------------------

    async def _plan(self, project_id: int, project_ctx: ProjectContext, replan: bool) -> None:
        self.events.emit("PLANNING_STARTED", project_id=project_id, payload={"replan": replan})
        extra = ""
        if replan:
            existing = [
                {
                    "task_key": t["task_key"],
                    "title": t["title"],
                    "status": t["status"],
                    "sequence": t["sequence"],
                    "dependencies": json.loads(t["dependencies"] or "[]"),
                }
                for t in self.tasks.list_for_project(project_id)
            ]
            extra = render_plan_for_replan(existing)

        plan: ProjectPlan = await self.invoker.invoke(
            planner.SPEC,
            project_id,
            project_ctx,
            extra=extra,
            artifact_path=self.artifacts.project_plan_path(),
        )
        terminal = {TaskState.COMPLETED.value, TaskState.SKIPPED.value}
        created, updated = [], []
        for planned in plan.tasks:
            existing = self.tasks.get_by_key(project_id, planned.task_key)
            if existing is None:
                self.tasks.create(
                    project_id,
                    planned.task_key,
                    planned.title,
                    planned.goal,
                    planned.acceptance_criteria,
                    planned.dependencies,
                )
                created.append(planned.task_key)
            elif existing["status"] not in terminal:
                # A revised plan replaces the definition of unfinished work;
                # completed/skipped tasks are never re-planned.
                self.tasks.update_definition(
                    existing["id"],
                    planned.title,
                    planned.goal,
                    planned.acceptance_criteria,
                    planned.dependencies,
                )
                if existing["status"] == TaskState.BLOCKED.value:
                    self.tasks.set_status(existing["id"], TaskState.PENDING, force=True)
                updated.append(planned.task_key)
        dropped = []
        if replan:
            # Unfinished tasks the revised plan no longer mentions are dropped,
            # not silently retried with their stale definition.
            plan_keys = {t.task_key for t in plan.tasks}
            for task in self.tasks.list_for_project(project_id):
                if task["status"] not in terminal and task["task_key"] not in plan_keys:
                    self.tasks.set_status(task["id"], TaskState.SKIPPED, force=True)
                    dropped.append(task["task_key"])
        self.events.emit(
            "PLAN_CREATED", project_id=project_id,
            payload={"created": created, "updated": updated, "dropped": dropped, "replan": replan},
        )
        logger.info(
            "plan %s: %d created, %d updated, %d dropped",
            "revised" if replan else "created", len(created), len(updated), len(dropped),
        )
