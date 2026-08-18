"""Append-only durable event ledger.

Events serve two purposes:
- audit trail (as before), and
- durable execution history for recovery, reconciliation, debugging and
  invariant validation.

Every event belongs to exactly one stream — (stream_type, stream_id) —
and receives a contiguous per-stream sequence number assigned inside the
same transaction as the INSERT, so the total order of a stream is fully
determined. The harness is the single writer, and a unique index on
(stream_type, stream_id, seq) makes gaps or duplicates impossible to
persist silently.

Normalized tables (projects/tasks/task_attempts/agent_runs/evaluations)
remain the fast-read source of current state; the ledger is history.
"""

from __future__ import annotations

import json
import sqlite3
from enum import StrEnum

from .connection import Database, utcnow


class StreamType(StrEnum):
    PROJECT = "project"
    TASK = "task"
    GLOBAL = "global"


class EventType(StrEnum):
    # project lifecycle
    PROJECT_CREATED = "PROJECT_CREATED"
    PROJECT_STATE_CHANGED = "PROJECT_STATE_CHANGED"
    PROJECT_IDENTITY_CORRECTED = "PROJECT_IDENTITY_CORRECTED"
    PROJECT_GOAL_UPDATED = "PROJECT_GOAL_UPDATED"
    PROJECT_PAUSED = "PROJECT_PAUSED"
    PROJECT_BLOCKED = "PROJECT_BLOCKED"
    PROJECT_FAILED = "PROJECT_FAILED"
    PROJECT_COMPLETED = "PROJECT_COMPLETED"
    BASELINE_COMMITTED = "BASELINE_COMMITTED"
    PLANNING_STARTED = "PLANNING_STARTED"
    PLAN_CREATED = "PLAN_CREATED"
    FINAL_VERIFICATION_STARTED = "FINAL_VERIFICATION_STARTED"
    # task lifecycle
    TASK_STARTED = "TASK_STARTED"
    TASK_STATE_CHANGED = "TASK_STATE_CHANGED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_BLOCKED = "TASK_BLOCKED"
    TASK_SPLIT = "TASK_SPLIT"
    TASK_REQUEUED = "TASK_REQUEUED"
    ATTEMPT_STARTED = "ATTEMPT_STARTED"
    ATTEMPT_FINISHED = "ATTEMPT_FINISHED"
    ATTEMPT_INTERRUPTED = "ATTEMPT_INTERRUPTED"
    # agent runs
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    PROVIDER_TELEMETRY = "PROVIDER_TELEMETRY"
    LOOP_WARNING = "LOOP_WARNING"
    LOOP_DETECTED = "LOOP_DETECTED"
    # verification / review / diagnosis
    TEST_STARTED = "TEST_STARTED"
    TEST_FAILED = "TEST_FAILED"
    REVIEW_STARTED = "REVIEW_STARTED"
    REVIEW_REPAIR = "REVIEW_REPAIR"
    REVIEW_REPLAN = "REVIEW_REPLAN"
    DIAGNOSIS_STARTED = "DIAGNOSIS_STARTED"
    DIAGNOSIS_COMPLETED = "DIAGNOSIS_COMPLETED"
    # operation intent/result journal
    AGENT_DISPATCH_INTENT = "AGENT_DISPATCH_INTENT"
    AGENT_DISPATCH_RESULT = "AGENT_DISPATCH_RESULT"
    VERIFICATION_COMMAND_INTENT = "VERIFICATION_COMMAND_INTENT"
    VERIFICATION_COMMAND_RESULT = "VERIFICATION_COMMAND_RESULT"
    GIT_COMMIT_INTENT = "GIT_COMMIT_INTENT"
    GIT_COMMIT_RESULT = "GIT_COMMIT_RESULT"
    GIT_COMMIT_RECONCILED = "GIT_COMMIT_RECONCILED"
    GIT_INTEGRATION_INTENT = "GIT_INTEGRATION_INTENT"
    GIT_INTEGRATION_RESULT = "GIT_INTEGRATION_RESULT"
    GIT_INTEGRATION_RECONCILED = "GIT_INTEGRATION_RECONCILED"
    INTEGRATION_STARTED = "INTEGRATION_STARTED"
    INTEGRATION_COMPLETED = "INTEGRATION_COMPLETED"
    INTEGRATION_CONFLICT = "INTEGRATION_CONFLICT"
    OPERATION_INTERRUPTED = "OPERATION_INTERRUPTED"
    METRICS_SNAPSHOT = "METRICS_SNAPSHOT"
    # recovery / invariants
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    INVARIANT_WARNING = "INVARIANT_WARNING"


class EventRepository:
    def __init__(self, db: Database):
        self.db = db

    def emit(
        self,
        event_type: EventType | str,
        project_id: int | None = None,
        task_id: int | None = None,
        attempt_id: int | None = None,
        payload: dict | None = None,
        agent_run_id: int | None = None,
        operation_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> int:
        """Append one event to its stream, with a contiguous seq number.

        The MAX(seq)+1 read and the INSERT run inside one transaction (the
        caller's, if already open), so a mid-append failure rolls back
        cleanly and never leaves a partial or gapped stream.
        """
        stream_type, stream_id = self._stream_for(project_id, task_id)
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM events "
                "WHERE stream_type = ? AND stream_id = ?",
                (stream_type.value, stream_id),
            )
            seq = row["next_seq"]
            cur = self.db.execute(
                """
                INSERT INTO events (project_id, task_id, attempt_id, agent_run_id,
                                    stream_type, stream_id, seq, event_type, payload,
                                    operation_id, correlation_id, causation_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    task_id,
                    attempt_id,
                    agent_run_id,
                    stream_type.value,
                    stream_id,
                    seq,
                    str(event_type),
                    json.dumps(payload or {}, ensure_ascii=False),
                    operation_id,
                    correlation_id,
                    causation_id,
                    utcnow(),
                ),
            )
            return cur.lastrowid

    @staticmethod
    def _stream_for(project_id: int | None, task_id: int | None) -> tuple[StreamType, int]:
        if task_id is not None:
            return StreamType.TASK, task_id
        if project_id is not None:
            return StreamType.PROJECT, project_id
        return StreamType.GLOBAL, 0

    # -- queries -----------------------------------------------------------

    def list_for_project(self, project_id: int, limit: int = 100) -> list[sqlite3.Row]:
        return self.db.query_all(
            "SELECT * FROM events WHERE project_id = ? ORDER BY id DESC LIMIT ?",
            (project_id, limit),
        )

    def list_stream(self, stream_type: StreamType | str, stream_id: int) -> list[sqlite3.Row]:
        return self.db.query_all(
            "SELECT * FROM events WHERE stream_type = ? AND stream_id = ? ORDER BY seq",
            (str(stream_type), stream_id),
        )

    def find_stream_gaps(self) -> list[dict]:
        """Streams whose seq numbering is not 1..N contiguous.

        Rows written before the ledger migration have seq NULL and are
        excluded — they were audit-only and carry no ordering guarantee.
        """
        rows = self.db.query_all(
            """
            SELECT stream_type, stream_id,
                   COUNT(*) AS n, MIN(seq) AS min_seq, MAX(seq) AS max_seq
            FROM events
            WHERE seq IS NOT NULL
            GROUP BY stream_type, stream_id
            HAVING COUNT(*) != MAX(seq) - MIN(seq) + 1 OR MIN(seq) != 1
            """
        )
        return [
            {
                "stream_type": row["stream_type"],
                "stream_id": row["stream_id"],
                "count": row["n"],
                "min_seq": row["min_seq"],
                "max_seq": row["max_seq"],
            }
            for row in rows
        ]
