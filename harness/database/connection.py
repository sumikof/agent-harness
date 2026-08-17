"""SQLite connection management.

Only the harness touches the database — agents never do. Sequential
execution keeps concurrency needs minimal; WAL + busy_timeout is enough.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from .migrations import apply_migrations


def utcnow() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._tx_depth = 0
        apply_migrations(self.conn)

    @contextlib.contextmanager
    def transaction(self, immediate: bool = False):
        """Group several execute() calls into one atomic SQLite transaction.

        Nested use joins the outermost transaction; commit happens only when
        the outermost block exits cleanly, rollback when it raises. This is
        how a state UPDATE and its ledger event INSERT stay atomic.

        immediate=True takes the write lock up front (BEGIN IMMEDIATE), so a
        read-then-insert sequence (e.g. the RUNNING-agent check before run
        creation) is serialized against other processes, not just other
        statements.
        """
        if immediate and self._tx_depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        self._tx_depth += 1
        try:
            yield self
        except BaseException:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.rollback()
            raise
        else:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.commit()

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        if self._tx_depth == 0:
            self.conn.commit()
        return cur

    def integrity_check(self) -> str:
        """Returns 'ok' when SQLite reports a healthy database file."""
        row = self.conn.execute("PRAGMA integrity_check").fetchone()
        return row[0] if row else "unknown"

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def query_all(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
