"""agent_runs and evaluations table access."""

from __future__ import annotations

import json
import sqlite3

from .connection import Database, utcnow


class RunRepository:
    def __init__(self, db: Database):
        self.db = db

    def start_run(
        self,
        project_id: int,
        role: str,
        attempt_id: int | None = None,
        input_artifact: str | None = None,
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO agent_runs (attempt_id, project_id, role, status, input_artifact, started_at)
            VALUES (?, ?, ?, 'RUNNING', ?, ?)
            """,
            (attempt_id, project_id, role, input_artifact, utcnow()),
        )
        return cur.lastrowid

    def finish_run(
        self,
        run_id: int,
        status: str,
        session_id: str | None = None,
        output_artifact: str | None = None,
        token_usage: dict | None = None,
        cost_usd: float = 0.0,
        error: str | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE agent_runs
            SET status = ?, session_id = ?, output_artifact = ?, token_usage = ?,
                cost_usd = ?, error = ?, finished_at = ?
            WHERE id = ?
            """,
            (
                status,
                session_id,
                output_artifact,
                json.dumps(token_usage or {}, ensure_ascii=False),
                cost_usd,
                error,
                utcnow(),
                run_id,
            ),
        )

    def count_runs_for_task(self, task_id: int) -> int:
        row = self.db.query_one(
            """
            SELECT COUNT(*) AS n FROM agent_runs r
            JOIN task_attempts a ON a.id = r.attempt_id
            WHERE a.task_id = ?
            """,
            (task_id,),
        )
        return row["n"]

    def record_evaluation(
        self, attempt_id: int, verdict: str, result: dict, score: float | None = None
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO evaluations (attempt_id, verdict, score, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (attempt_id, verdict, score, json.dumps(result, ensure_ascii=False), utcnow()),
        )
        return cur.lastrowid

    def list_runs_for_attempt(self, attempt_id: int) -> list[sqlite3.Row]:
        return self.db.query_all(
            "SELECT * FROM agent_runs WHERE attempt_id = ? ORDER BY id", (attempt_id,)
        )
