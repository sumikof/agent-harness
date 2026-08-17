"""Project Loop: plan once, then run tasks strictly one at a time.

Owns wiring of every component and the project-level state machine.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from ..agents import planner
from ..artifacts.manager import ArtifactManager
from ..artifacts.schemas import ProjectPlan
from ..config import HarnessConfig
from ..context.builder import ContextBuilder, render_plan_for_replan
from ..context.project_context import DEFAULT_FORBIDDEN_OPERATIONS, ProjectContext
from ..database.connection import Database
from ..database.event_repository import EventRepository
from ..database.project_repository import ProjectRepository
from ..database.run_repository import RunRepository
from ..database.task_repository import TaskRepository
from ..git.checkpoint import CheckpointManager
from ..git.repository import GitRepository
from ..verification.runner import VerificationRunner
from .agent_invoker import AgentInvoker, AgentRunFailed
from .budget import BudgetExceeded, BudgetManager
from .recovery import RecoveryManager
from .state_machine import ProjectState, TaskState
from .task_runner import TaskOutcome, TaskRunner

logger = logging.getLogger(__name__)

MAX_REPLANS = 3


class ProjectOrchestrator:
    def __init__(self, config: HarnessConfig):
        self.config = config
        config.workspace_path.mkdir(parents=True, exist_ok=True)
        config.logs_path.mkdir(parents=True, exist_ok=True)

        self.db = Database(config.db_path)
        self.projects = ProjectRepository(self.db)
        self.tasks = TaskRepository(self.db)
        self.runs = RunRepository(self.db)
        self.events = EventRepository(self.db)
        self.artifacts = ArtifactManager(config.artifacts_path)
        self.git = GitRepository(config.repository_path)
        self.checkpoint = CheckpointManager(self.git)
        self.verifier = VerificationRunner(
            config.verification, config.repository_path, config.logs_path / "verification"
        )
        self.context_builder = ContextBuilder(config.prompts_path)
        self.budget = BudgetManager(config, self.projects, self.tasks, self.runs)
        self.invoker = AgentInvoker(
            config, self.context_builder, self.artifacts, self.runs, self.events, self.budget
        )
        self.task_runner = TaskRunner(
            config, self.invoker, self.tasks, self.runs, self.events,
            self.artifacts, self.git, self.checkpoint, self.verifier,
        )
        self.recovery = RecoveryManager(
            self.tasks, self.events, self.artifacts, self.git, self.checkpoint
        )

    # ------------------------------------------------------------------

    def ensure_project(self) -> sqlite3.Row:
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
        if not self.git.is_repo():
            raise RuntimeError(
                f"{self.config.repository_path} is not a git repository. "
                "Place or clone the target repository there first."
            )
        return row

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
        self.recovery.recover(project)
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

            if state in (ProjectState.READY, ProjectState.PAUSED, ProjectState.BLOCKED,
                         ProjectState.PLANNING, ProjectState.REPLANNING):
                self.projects.set_status(project_id, ProjectState.RUNNING, force=True)

            return await self._run_task_loop(project_id, project_ctx)
        except BudgetExceeded as exc:
            logger.warning("project paused: %s", exc)
            self.projects.set_status(project_id, ProjectState.PAUSED, force=True)
            self.events.emit("PROJECT_PAUSED", project_id=project_id, payload={"reason": str(exc)})
            return ProjectState.PAUSED
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
        blocked = [
            t["task_key"]
            for t in self.tasks.list_for_project(project_id)
            if t["status"] == TaskState.BLOCKED.value
        ]
        if blocked:
            self.projects.set_status(project_id, ProjectState.BLOCKED, force=True)
            self.events.emit("PROJECT_BLOCKED", project_id=project_id,
                             payload={"blocked_tasks": blocked})
            logger.warning("project blocked; blocked tasks: %s", ", ".join(blocked))
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
        inserted = []
        for planned in plan.tasks:
            if self.tasks.get_by_key(project_id, planned.task_key) is not None:
                continue  # completed and existing tasks are never re-planned
            self.tasks.create(
                project_id,
                planned.task_key,
                planned.title,
                planned.goal,
                planned.acceptance_criteria,
                planned.dependencies,
            )
            inserted.append(planned.task_key)
        self.events.emit("PLAN_CREATED", project_id=project_id,
                         payload={"tasks": inserted, "replan": replan})
        logger.info("plan %s: %d new task(s)", "revised" if replan else "created", len(inserted))
