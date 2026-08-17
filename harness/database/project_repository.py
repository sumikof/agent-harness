"""Project table access."""

from __future__ import annotations

import sqlite3

from ..orchestrator.state_machine import ProjectState, assert_project_transition
from .connection import Database, utcnow


class ProjectRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(
        self,
        name: str,
        repository: str,
        base_branch: str,
        goal: str,
        budget_usd: float,
    ) -> int:
        now = utcnow()
        cur = self.db.execute(
            """
            INSERT INTO projects (name, repository, base_branch, goal, status, budget_usd, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, repository, base_branch, goal, ProjectState.CREATED.value, budget_usd, now, now),
        )
        return cur.lastrowid

    def get(self, project_id: int) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM projects WHERE id = ?", (project_id,))

    def get_by_name(self, name: str) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM projects WHERE name = ?", (name,))

    def set_status(self, project_id: int, status: ProjectState, *, force: bool = False) -> None:
        row = self.get(project_id)
        if row is None:
            raise ValueError(f"unknown project id {project_id}")
        if not force:
            assert_project_transition(ProjectState(row["status"]), status)
        self.db.execute(
            "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, utcnow(), project_id),
        )

    def update_goal(self, project_id: int, goal: str) -> None:
        self.db.execute(
            "UPDATE projects SET goal = ?, updated_at = ? WHERE id = ?",
            (goal, utcnow(), project_id),
        )

    def add_spent(self, project_id: int, cost_usd: float) -> float:
        self.db.execute(
            "UPDATE projects SET spent_usd = spent_usd + ?, updated_at = ? WHERE id = ?",
            (cost_usd, utcnow(), project_id),
        )
        row = self.get(project_id)
        return row["spent_usd"]
