"""Append-only event log."""

from __future__ import annotations

import json
import sqlite3

from .connection import Database, utcnow


class EventRepository:
    def __init__(self, db: Database):
        self.db = db

    def emit(
        self,
        event_type: str,
        project_id: int | None = None,
        task_id: int | None = None,
        attempt_id: int | None = None,
        payload: dict | None = None,
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO events (project_id, task_id, attempt_id, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                task_id,
                attempt_id,
                event_type,
                json.dumps(payload or {}, ensure_ascii=False),
                utcnow(),
            ),
        )
        return cur.lastrowid

    def list_for_project(self, project_id: int, limit: int = 100) -> list[sqlite3.Row]:
        return self.db.query_all(
            "SELECT * FROM events WHERE project_id = ? ORDER BY id DESC LIMIT ?",
            (project_id, limit),
        )
