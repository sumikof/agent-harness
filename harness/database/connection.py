"""SQLite connection management.

Only the harness touches the database — agents never do. The harness runs
many agent coroutines in parallel (and pushes blocking work like
verification onto worker threads), so the old "sequential execution keeps
concurrency needs minimal" assumption is gone. Instead:

    Parallel agent tasks / worker threads
                   │
                   ▼
        single-writer serialization
        (one process-wide RLock around
         every statement + transaction)
                   │
                   ▼
                SQLite (WAL)

One connection, one writer: transactions from different coroutines or
threads never interleave, so the check-then-insert patterns and the
"state UPDATE + ledger event INSERT in one transaction" atomicity hold
exactly as they did under sequential execution. WAL + busy_timeout remain
for cross-process readers (status/events CLI).
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path

from .migrations import apply_migrations


def utcnow() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Database:
    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        journal_mode: str = "WAL",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: blocking work (verification, git) runs on
        # worker threads and may record progress. Safe because EVERY access
        # goes through the single-writer lock below.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA journal_mode={journal_mode}")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        # Reentrant so nested transaction() blocks (already the established
        # pattern) keep working; held for the whole outermost transaction so
        # no other coroutine/thread can interleave statements into it.
        self._writer_lock = threading.RLock()
        self._tx_depth = 0
        apply_migrations(self.conn)

    @contextlib.contextmanager
    def transaction(self, immediate: bool = False):
        """Group several execute() calls into one atomic SQLite transaction.

        Nested use joins the outermost transaction; commit happens only when
        the outermost block exits cleanly, rollback when it raises. This is
        how a state UPDATE and its ledger event INSERT stay atomic.

        The single-writer lock is held for the entire outermost transaction,
        serializing it against every other statement in this process —
        parallel agent tasks cannot interleave partial writes.

        immediate=True takes the SQLite write lock up front (BEGIN
        IMMEDIATE), serializing the read-then-insert against other
        PROCESSES as well (the in-process case is already covered by the
        writer lock).
        """
        with self._writer_lock:
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
        with self._writer_lock:
            cur = self.conn.execute(sql, params)
            if self._tx_depth == 0:
                self.conn.commit()
            return cur

    def integrity_check(self) -> str:
        """Returns 'ok' when SQLite reports a healthy database file."""
        with self._writer_lock:
            row = self.conn.execute("PRAGMA integrity_check").fetchone()
        return row[0] if row else "unknown"

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        with self._writer_lock:
            return self.conn.execute(sql, params).fetchone()

    def query_all(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._writer_lock:
            return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._writer_lock:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
