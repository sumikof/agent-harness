"""SQLite single-writer serialization under real thread concurrency
(spec items 34-37, 71)."""

import threading

from harness.database.connection import Database
from harness.database.event_repository import EventRepository


def test_concurrent_transactions_serialize_without_corruption(tmp_path):
    db = Database(tmp_path / "test.db")
    db.execute("CREATE TABLE counters (id INTEGER PRIMARY KEY, value INTEGER)")
    db.execute("INSERT INTO counters (id, value) VALUES (1, 0)")
    errors: list[Exception] = []

    def bump(times: int) -> None:
        try:
            for _ in range(times):
                with db.transaction():
                    row = db.query_one("SELECT value FROM counters WHERE id = 1")
                    db.execute("UPDATE counters SET value = ? WHERE id = 1",
                               (row["value"] + 1,))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=bump, args=(50,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # read-modify-write x 400 with zero lost updates == fully serialized
    assert db.query_one("SELECT value FROM counters WHERE id = 1")["value"] == 400
    assert db.integrity_check() == "ok"
    db.close()


def test_event_seq_contiguous_per_stream_under_parallel_emit(tmp_path):
    """Parallel emitters across MANY streams: every stream must still get a
    contiguous 1..N seq — no gaps, no duplicates."""
    db = Database(tmp_path / "test.db")
    events = EventRepository(db)
    now = "2026-01-01T00:00:00Z"
    db.execute(
        "INSERT INTO projects (id, name, repository, created_at, updated_at) "
        "VALUES (1, 'p', '/r', ?, ?)", (now, now))
    for task_id in range(1, 9):
        db.execute(
            "INSERT INTO tasks (id, task_key, project_id, sequence, title, "
            "created_at, updated_at) VALUES (?, ?, 1, ?, 't', ?, ?)",
            (task_id, f"T{task_id:03d}", task_id * 10, now, now))
    errors: list[Exception] = []

    def emit_for_task(task_id: int, count: int) -> None:
        try:
            for _ in range(count):
                events.emit("TASK_STATE_CHANGED", project_id=1, task_id=task_id,
                            payload={"n": task_id})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=emit_for_task, args=(task_id, 30))
               for task_id in range(1, 9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert events.find_stream_gaps() == []
    for task_id in range(1, 9):
        rows = events.list_stream("task", task_id)
        assert [r["seq"] for r in rows] == list(range(1, 31))
    db.close()


def test_nested_transaction_joins_outermost(tmp_path):
    db = Database(tmp_path / "test.db")
    db.execute("CREATE TABLE t (x INTEGER)")
    try:
        with db.transaction():
            db.execute("INSERT INTO t (x) VALUES (1)")
            with db.transaction():
                db.execute("INSERT INTO t (x) VALUES (2)")
            raise RuntimeError("abort outermost")
    except RuntimeError:
        pass
    # everything rolled back together
    assert db.query_one("SELECT COUNT(*) AS n FROM t")["n"] == 0
    db.close()
