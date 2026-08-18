"""agent_runs and evaluations table access."""

from __future__ import annotations

import json
import sqlite3

from .connection import Database, utcnow

# evaluations.verdict values recorded by the harness itself (deterministic
# verification evidence), alongside the Reviewer's PASS/REPAIR/REPLAN.
VERIFY_PASS = "VERIFY_PASS"
VERIFY_FAIL = "VERIFY_FAIL"


class RunRepository:
    def __init__(self, db: Database):
        self.db = db

    def start_run(
        self,
        project_id: int,
        role: str,
        attempt_id: int | None = None,
        input_artifact: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        profile_hash: str | None = None,
        profile_version: str | None = None,
        context_manifest_path: str | None = None,
        context_manifest_hash: str | None = None,
        resolved_spec: dict | None = None,
        dispatch_operation_id: str | None = None,
        mutating: bool = False,
        prefix_group_key: str | None = None,
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO agent_runs (attempt_id, project_id, role, status, input_artifact,
                                    started_at, provider, model, profile_hash, profile_version,
                                    context_manifest_path, context_manifest_hash,
                                    resolved_spec, dispatch_operation_id,
                                    mutating, prefix_group_key)
            VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                project_id,
                role,
                input_artifact,
                utcnow(),
                provider,
                model,
                profile_hash,
                profile_version,
                context_manifest_path,
                context_manifest_hash,
                json.dumps(resolved_spec, ensure_ascii=False) if resolved_spec else None,
                dispatch_operation_id,
                1 if mutating else 0,
                prefix_group_key,
            ),
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

    def get(self, run_id: int) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM agent_runs WHERE id = ?", (run_id,))

    def running_runs(self, project_id: int | None = None) -> list[sqlite3.Row]:
        if project_id is None:
            return self.db.query_all("SELECT * FROM agent_runs WHERE status = 'RUNNING'")
        return self.db.query_all(
            "SELECT * FROM agent_runs WHERE status = 'RUNNING' AND project_id = ?",
            (project_id,),
        )

    def running_mutating_for_attempt(self, attempt_id: int) -> list[sqlite3.Row]:
        return self.db.query_all(
            "SELECT * FROM agent_runs WHERE status = 'RUNNING' AND mutating = 1 "
            "AND attempt_id = ?",
            (attempt_id,),
        )

    def interrupt_running(self, project_id: int | None = None) -> list[int]:
        """Close RUNNING run rows left behind by a crash. Returns their ids.

        Recovery passes None: the workspace lock guarantees no other live
        process, so ANY remaining RUNNING row — regardless of project — is a
        crash leftover, and leaving it would consume slots of the bounded
        parallel-agent-run budget for every project sharing the DB.
        """
        rows = self.running_runs(project_id)
        for row in rows:
            self.finish_run(row["id"], "INTERRUPTED", error="interrupted (recovered at startup)")
        return [row["id"] for row in rows]

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

    def has_evaluation(self, attempt_id: int, verdict: str) -> bool:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM evaluations WHERE attempt_id = ? AND verdict = ?",
            (attempt_id, verdict),
        )
        return row["n"] > 0

    def list_runs_for_attempt(self, attempt_id: int) -> list[sqlite3.Row]:
        return self.db.query_all(
            "SELECT * FROM agent_runs WHERE attempt_id = ? ORDER BY id", (attempt_id,)
        )
