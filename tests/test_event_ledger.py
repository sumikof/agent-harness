"""Durable event ledger: contiguous streams, atomicity, migration safety."""

import pytest

import harness.database.migrations as migrations_module
from harness.database.connection import Database
from harness.database.event_repository import EventRepository, EventType
from harness.database.project_repository import ProjectRepository
from harness.database.task_repository import TaskRepository
from harness.orchestrator.state_machine import TaskState


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "harness.db")
    yield database
    database.close()


def test_event_seq_is_contiguous_per_stream(db):
    events = EventRepository(db)
    projects = ProjectRepository(db, events)
    pid = projects.create("p", "/repo", "main", "g", 10.0)
    tasks = TaskRepository(db, events)
    t1 = tasks.create(pid, "T001", "one")
    t2 = tasks.create(pid, "T002", "two")

    for _ in range(3):
        events.emit(EventType.TASK_STARTED, project_id=pid, task_id=t1)
    events.emit(EventType.TASK_STARTED, project_id=pid, task_id=t2)
    events.emit(EventType.PLANNING_STARTED, project_id=pid)

    stream_t1 = events.list_stream("task", t1)
    assert [row["seq"] for row in stream_t1] == [1, 2, 3]
    stream_t2 = events.list_stream("task", t2)
    assert [row["seq"] for row in stream_t2] == [1]
    assert events.find_stream_gaps() == []


def test_event_append_rolls_back_with_failed_transaction(db):
    events = EventRepository(db)
    events.emit(EventType.PLANNING_STARTED, project_id=None)  # global stream seq 1

    with pytest.raises(RuntimeError):
        with db.transaction():
            events.emit(EventType.PLANNING_STARTED)
            events.emit(EventType.PLANNING_STARTED)
            raise RuntimeError("boom")

    rows = events.list_stream("global", 0)
    assert [row["seq"] for row in rows] == [1]  # both rolled back, no gap
    assert events.find_stream_gaps() == []


def test_task_state_change_and_event_are_atomic(db, monkeypatch):
    events = EventRepository(db)
    projects = ProjectRepository(db, events)
    pid = projects.create("p", "/repo", "main", "g", 10.0)
    tasks = TaskRepository(db, events)
    t1 = tasks.create(pid, "T001", "one")

    tasks.set_status(t1, TaskState.READY)
    stream = events.list_stream("task", t1)
    assert stream[-1]["event_type"] == "TASK_STATE_CHANGED"

    # If the event INSERT fails, the state UPDATE must roll back with it.
    original_emit = events.emit

    def failing_emit(*args, **kwargs):
        raise RuntimeError("ledger write failed")

    monkeypatch.setattr(events, "emit", failing_emit)
    with pytest.raises(RuntimeError):
        tasks.set_status(t1, TaskState.ANALYZING)
    monkeypatch.setattr(events, "emit", original_emit)

    assert tasks.get(t1)["status"] == "READY"  # UPDATE rolled back too


def test_stream_gap_detection(db):
    events = EventRepository(db)
    events.emit(EventType.PLANNING_STARTED)
    events.emit(EventType.PLANNING_STARTED)
    # forge a gap directly (bypassing the repository)
    db.execute("UPDATE events SET seq = 5 WHERE seq = 2")
    gaps = events.find_stream_gaps()
    assert len(gaps) == 1
    assert gaps[0]["stream_type"] == "global"


def test_migration_preserves_v1_workspace(tmp_path, monkeypatch):
    """A workspace created before the ledger migration must open, keep its
    data, and accept new ledger events afterwards."""
    db_path = tmp_path / "harness.db"

    monkeypatch.setattr(migrations_module, "MIGRATIONS", migrations_module.MIGRATIONS[:1])
    old_db = Database(db_path)
    old_db.execute(
        "INSERT INTO projects (name, repository, base_branch, goal, status, budget_usd, "
        "created_at, updated_at) VALUES ('p', '/repo', 'main', 'g', 'CREATED', 10, 't', 't')"
    )
    old_db.execute(
        "INSERT INTO events (project_id, event_type, payload, created_at) "
        "VALUES (1, 'PROJECT_CREATED', '{}', 't')"
    )
    old_db.close()

    monkeypatch.undo()
    new_db = Database(db_path)  # applies migration 2
    try:
        row = new_db.query_one("SELECT * FROM projects WHERE name = 'p'")
        assert row is not None and row["status"] == "CREATED"
        legacy = new_db.query_one("SELECT * FROM events WHERE event_type = 'PROJECT_CREATED'")
        assert legacy["seq"] is None  # legacy rows carry no ordering claim

        events = EventRepository(new_db)
        events.emit(EventType.TASK_STARTED, project_id=1)
        assert events.find_stream_gaps() == []  # legacy NULL rows are exempt
        new = new_db.query_one("SELECT * FROM events WHERE event_type = 'TASK_STARTED'")
        assert new["seq"] == 1 and new["stream_type"] == "project"
        assert new_db.query_one("SELECT 1 FROM operations LIMIT 1") is None  # table exists
    finally:
        new_db.close()
