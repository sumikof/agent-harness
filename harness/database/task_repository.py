"""Task and task_attempt table access."""

from __future__ import annotations

import json
import sqlite3

from ..orchestrator.state_machine import AttemptState, TaskState, assert_task_transition
from .connection import Database, utcnow

# Sequence numbers are spaced so tasks can be inserted between existing ones
# (10, 20, 30 ... then 25 for an inserted T00XA).
SEQUENCE_STEP = 10


class TaskRepository:
    def __init__(self, db: Database):
        self.db = db

    # -- tasks -------------------------------------------------------------

    def create(
        self,
        project_id: int,
        task_key: str,
        title: str,
        goal: str = "",
        acceptance_criteria: list[str] | None = None,
        dependencies: list[str] | None = None,
        sequence: int | None = None,
    ) -> int:
        now = utcnow()
        if sequence is None:
            row = self.db.query_one(
                "SELECT MAX(sequence) AS max_seq FROM tasks WHERE project_id = ?", (project_id,)
            )
            sequence = (row["max_seq"] or 0) + SEQUENCE_STEP
        cur = self.db.execute(
            """
            INSERT INTO tasks (task_key, project_id, sequence, title, goal,
                               acceptance_criteria, dependencies, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_key,
                project_id,
                sequence,
                title,
                goal,
                json.dumps(acceptance_criteria or [], ensure_ascii=False),
                json.dumps(dependencies or [], ensure_ascii=False),
                TaskState.PENDING.value,
                now,
                now,
            ),
        )
        return cur.lastrowid

    def get(self, task_id: int) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))

    def get_by_key(self, project_id: int, task_key: str) -> sqlite3.Row | None:
        return self.db.query_one(
            "SELECT * FROM tasks WHERE project_id = ? AND task_key = ?", (project_id, task_key)
        )

    def list_for_project(self, project_id: int) -> list[sqlite3.Row]:
        return self.db.query_all(
            "SELECT * FROM tasks WHERE project_id = ? ORDER BY sequence", (project_id,)
        )

    def next_runnable(self, project_id: int) -> sqlite3.Row | None:
        """Next PENDING/READY task (by sequence) whose dependencies are all COMPLETED."""
        candidates = self.db.query_all(
            """
            SELECT * FROM tasks
            WHERE project_id = ? AND status IN ('PENDING', 'READY')
            ORDER BY sequence
            """,
            (project_id,),
        )
        completed = {
            row["task_key"]
            for row in self.db.query_all(
                "SELECT task_key FROM tasks WHERE project_id = ? AND status IN ('COMPLETED', 'SKIPPED')",
                (project_id,),
            )
        }
        for task in candidates:
            deps = json.loads(task["dependencies"] or "[]")
            if all(dep in completed for dep in deps):
                return task
        return None

    def has_unfinished(self, project_id: int) -> bool:
        row = self.db.query_one(
            """
            SELECT COUNT(*) AS n FROM tasks
            WHERE project_id = ? AND status NOT IN ('COMPLETED', 'SKIPPED', 'BLOCKED')
            """,
            (project_id,),
        )
        return row["n"] > 0

    def set_status(self, task_id: int, status: TaskState, *, force: bool = False) -> None:
        row = self.get(task_id)
        if row is None:
            raise ValueError(f"unknown task id {task_id}")
        if not force:
            assert_task_transition(TaskState(row["status"]), status)
        self.db.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, utcnow(), task_id),
        )

    def update_definition(
        self,
        task_id: int,
        title: str,
        goal: str,
        acceptance_criteria: list[str] | None,
        dependencies: list[str] | None,
    ) -> None:
        """Replace an unfinished task's definition with a revised plan's version."""
        self.db.execute(
            """
            UPDATE tasks SET title = ?, goal = ?, acceptance_criteria = ?,
                             dependencies = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                goal,
                json.dumps(acceptance_criteria or [], ensure_ascii=False),
                json.dumps(dependencies or [], ensure_ascii=False),
                utcnow(),
                task_id,
            ),
        )

    def set_commit(self, task_id: int, commit_hash: str) -> None:
        self.db.execute(
            "UPDATE tasks SET current_commit = ?, updated_at = ? WHERE id = ?",
            (commit_hash, utcnow(), task_id),
        )

    def add_spent(self, task_id: int, cost_usd: float) -> float:
        self.db.execute(
            "UPDATE tasks SET spent_usd = spent_usd + ?, updated_at = ? WHERE id = ?",
            (cost_usd, utcnow(), task_id),
        )
        return self.get(task_id)["spent_usd"]

    # -- attempts ----------------------------------------------------------

    def start_attempt(self, task_id: int, base_commit: str | None) -> int:
        row = self.db.query_one(
            "SELECT MAX(attempt_no) AS max_no FROM task_attempts WHERE task_id = ?", (task_id,)
        )
        attempt_no = (row["max_no"] or 0) + 1
        cur = self.db.execute(
            """
            INSERT INTO task_attempts (task_id, attempt_no, status, base_commit, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, attempt_no, AttemptState.RUNNING.value, base_commit, utcnow()),
        )
        self.db.execute(
            "UPDATE tasks SET attempt_count = ?, updated_at = ? WHERE id = ?",
            (attempt_no, utcnow(), task_id),
        )
        return cur.lastrowid

    def finish_attempt(self, attempt_id: int, status: AttemptState) -> None:
        self.db.execute(
            "UPDATE task_attempts SET status = ?, finished_at = ? WHERE id = ?",
            (status.value, utcnow(), attempt_id),
        )

    def get_attempt(self, attempt_id: int) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM task_attempts WHERE id = ?", (attempt_id,))

    def running_attempts(self, project_id: int) -> list[sqlite3.Row]:
        return self.db.query_all(
            """
            SELECT a.* FROM task_attempts a
            JOIN tasks t ON t.id = a.task_id
            WHERE t.project_id = ? AND a.status = 'RUNNING'
            """,
            (project_id,),
        )

    def consecutive_failures(self, task_id: int) -> int:
        """Failed attempts since the last PASSED attempt."""
        rows = self.db.query_all(
            "SELECT status FROM task_attempts WHERE task_id = ? ORDER BY attempt_no DESC", (task_id,)
        )
        count = 0
        for row in rows:
            if row["status"] == AttemptState.FAILED.value:
                count += 1
            elif row["status"] == AttemptState.PASSED.value:
                break
        return count
