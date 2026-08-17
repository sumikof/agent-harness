"""Schema migrations, applied idempotently at startup."""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        repository TEXT NOT NULL,
        base_branch TEXT NOT NULL DEFAULT 'main',
        goal TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'CREATED',
        budget_usd REAL NOT NULL DEFAULT 0,
        spent_usd REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_key TEXT NOT NULL,
        project_id INTEGER NOT NULL REFERENCES projects(id),
        sequence INTEGER NOT NULL,
        title TEXT NOT NULL,
        goal TEXT NOT NULL DEFAULT '',
        acceptance_criteria TEXT NOT NULL DEFAULT '[]',
        dependencies TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'PENDING',
        attempt_count INTEGER NOT NULL DEFAULT 0,
        spent_usd REAL NOT NULL DEFAULT 0,
        current_commit TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (project_id, task_key)
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_project_seq ON tasks(project_id, sequence);

    CREATE TABLE IF NOT EXISTS task_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL REFERENCES tasks(id),
        attempt_no INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'RUNNING',
        base_commit TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        UNIQUE (task_id, attempt_no)
    );

    CREATE TABLE IF NOT EXISTS agent_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER REFERENCES task_attempts(id),
        project_id INTEGER NOT NULL REFERENCES projects(id),
        role TEXT NOT NULL,
        session_id TEXT,
        status TEXT NOT NULL DEFAULT 'RUNNING',
        input_artifact TEXT,
        output_artifact TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        token_usage TEXT,
        cost_usd REAL NOT NULL DEFAULT 0,
        error TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_agent_runs_attempt ON agent_runs(attempt_id);

    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER NOT NULL REFERENCES task_attempts(id),
        verdict TEXT NOT NULL,
        score REAL,
        result_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER REFERENCES projects(id),
        task_id INTEGER REFERENCES tasks(id),
        attempt_id INTEGER REFERENCES task_attempts(id),
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id, id);
    """,
]


def apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    from .connection import utcnow

    for version, sql in enumerate(MIGRATIONS, start=1):
        if version in applied:
            continue
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, utcnow()),
        )
    conn.commit()
