"""Operation intent/result journal for harness-owned side effects.

Any externally-visible side effect (agent dispatch, verification command,
git commit) is journaled as:

    intent persisted (durable COMMIT)  ->  side effect executed
                                       ->  result persisted

Intent and result share one operation_id, so after a crash the recovery
subsystem can enumerate unfinished operations and decide — by inspecting
the real world (e.g. the git log) — whether the side effect happened.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from enum import StrEnum

from .connection import Database, utcnow
from .event_repository import EventRepository, EventType


class OperationType(StrEnum):
    AGENT_DISPATCH = "AGENT_DISPATCH"
    VERIFICATION_COMMAND = "VERIFICATION_COMMAND"
    GIT_COMMIT = "GIT_COMMIT"


class OperationStatus(StrEnum):
    PENDING = "PENDING"          # intent persisted; side effect state unknown
    COMPLETED = "COMPLETED"      # side effect confirmed done
    FAILED = "FAILED"            # side effect confirmed NOT done / failed
    INTERRUPTED = "INTERRUPTED"  # crash window; side effect may or may not exist
    RECONCILED = "RECONCILED"    # side effect found post-crash; DB caught up


_INTENT_EVENTS: dict[OperationType, EventType] = {
    OperationType.AGENT_DISPATCH: EventType.AGENT_DISPATCH_INTENT,
    OperationType.VERIFICATION_COMMAND: EventType.VERIFICATION_COMMAND_INTENT,
    OperationType.GIT_COMMIT: EventType.GIT_COMMIT_INTENT,
}

_RESULT_EVENTS: dict[OperationType, EventType] = {
    OperationType.AGENT_DISPATCH: EventType.AGENT_DISPATCH_RESULT,
    OperationType.VERIFICATION_COMMAND: EventType.VERIFICATION_COMMAND_RESULT,
    OperationType.GIT_COMMIT: EventType.GIT_COMMIT_RESULT,
}


class OperationRepository:
    def __init__(self, db: Database, events: EventRepository | None = None):
        self.db = db
        self.events = events

    def record_intent(
        self,
        operation_type: OperationType,
        payload: dict,
        *,
        project_id: int | None = None,
        task_id: int | None = None,
        attempt_id: int | None = None,
        agent_run_id: int | None = None,
        operation_id: str | None = None,
    ) -> str:
        """Persist the intent (and its ledger event) atomically; returns operation_id."""
        operation_id = operation_id or uuid.uuid4().hex
        with self.db.transaction():
            self.db.execute(
                """
                INSERT INTO operations (operation_id, operation_type, status, project_id,
                                        task_id, attempt_id, agent_run_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    operation_type.value,
                    OperationStatus.PENDING.value,
                    project_id,
                    task_id,
                    attempt_id,
                    agent_run_id,
                    json.dumps(payload, ensure_ascii=False),
                    utcnow(),
                ),
            )
            if self.events:
                self.events.emit(
                    _INTENT_EVENTS[operation_type],
                    project_id=project_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    agent_run_id=agent_run_id,
                    operation_id=operation_id,
                    payload=payload,
                )
        return operation_id

    def record_result(
        self,
        operation_id: str,
        status: OperationStatus,
        result: dict | None = None,
    ) -> None:
        row = self.get(operation_id)
        if row is None:
            raise ValueError(f"unknown operation_id {operation_id}")
        operation_type = OperationType(row["operation_type"])
        with self.db.transaction():
            self.db.execute(
                "UPDATE operations SET status = ?, result = ?, completed_at = ? WHERE operation_id = ?",
                (
                    status.value,
                    json.dumps(result or {}, ensure_ascii=False),
                    utcnow(),
                    operation_id,
                ),
            )
            if self.events:
                event_type = (
                    EventType.OPERATION_INTERRUPTED
                    if status == OperationStatus.INTERRUPTED
                    else _RESULT_EVENTS[operation_type]
                )
                self.events.emit(
                    event_type,
                    project_id=row["project_id"],
                    task_id=row["task_id"],
                    attempt_id=row["attempt_id"],
                    agent_run_id=row["agent_run_id"],
                    operation_id=operation_id,
                    payload={"status": status.value, **(result or {})},
                )

    def get(self, operation_id: str) -> sqlite3.Row | None:
        return self.db.query_one(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
        )

    def unfinished(self, operation_type: OperationType | None = None) -> list[sqlite3.Row]:
        """Operations whose intent was persisted but no result was recorded."""
        if operation_type is None:
            return self.db.query_all(
                "SELECT * FROM operations WHERE status = ? ORDER BY id",
                (OperationStatus.PENDING.value,),
            )
        return self.db.query_all(
            "SELECT * FROM operations WHERE status = ? AND operation_type = ? ORDER BY id",
            (OperationStatus.PENDING.value, operation_type.value),
        )
