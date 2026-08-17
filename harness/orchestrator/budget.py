"""Budget enforcement.

Three layers: project budget, task budget, per-agent-run cap — plus hard
limits on turns / agent runs / wall clock held in config. Exceeding a
budget never crashes the harness; it transitions work to BLOCKED/PAUSED.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from ..config import HarnessConfig
from ..database.project_repository import ProjectRepository
from ..database.run_repository import RunRepository
from ..database.task_repository import TaskRepository


class BudgetExceeded(Exception):
    def __init__(self, scope: str, detail: str):
        self.scope = scope
        self.detail = detail
        super().__init__(f"budget exceeded ({scope}): {detail}")


class BudgetManager:
    def __init__(
        self,
        config: HarnessConfig,
        projects: ProjectRepository,
        tasks: TaskRepository,
        runs: RunRepository,
    ):
        self.config = config
        self.projects = projects
        self.tasks = tasks
        self.runs = runs

    def check_project(self, project_id: int) -> None:
        row = self.projects.get(project_id)
        if row["spent_usd"] >= self.config.budget.project_usd:
            raise BudgetExceeded(
                "project", f"spent ${row['spent_usd']:.2f} of ${self.config.budget.project_usd:.2f}"
            )
        # Wall clock is measured from the project's persisted creation time,
        # not process start — restarting the harness must not grant a fresh
        # max_execution_seconds interval.
        elapsed = self._project_elapsed_seconds(row)
        if elapsed > self.config.limits.max_execution_seconds:
            raise BudgetExceeded(
                "project",
                f"wall clock {elapsed:.0f}s exceeded {self.config.limits.max_execution_seconds}s "
                "(measured from project creation)",
            )

    @staticmethod
    def _project_elapsed_seconds(row: sqlite3.Row) -> float:
        try:
            started = datetime.fromisoformat(row["created_at"])
        except (ValueError, TypeError):
            return 0.0
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - started).total_seconds()

    def check_task(self, task_id: int) -> None:
        row = self.tasks.get(task_id)
        if row["spent_usd"] >= self.config.budget.task_usd:
            raise BudgetExceeded(
                "task", f"task {row['task_key']} spent ${row['spent_usd']:.2f} of ${self.config.budget.task_usd:.2f}"
            )
        if self.runs.count_runs_for_task(task_id) >= self.config.limits.max_agent_runs_per_task:
            raise BudgetExceeded(
                "task", f"task {row['task_key']} hit max_agent_runs_per_task={self.config.limits.max_agent_runs_per_task}"
            )

    def record_cost(self, project_id: int, task_id: int | None, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        self.projects.add_spent(project_id, cost_usd)
        if task_id is not None:
            self.tasks.add_spent(task_id, cost_usd)
