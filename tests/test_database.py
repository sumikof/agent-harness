import pytest

from harness.database.connection import Database
from harness.database.event_repository import EventRepository
from harness.database.project_repository import ProjectRepository
from harness.database.run_repository import RunRepository
from harness.database.task_repository import TaskRepository
from harness.orchestrator.state_machine import AttemptState, ProjectState, TaskState


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "harness.db")
    yield database
    database.close()


@pytest.fixture
def project_id(db):
    return ProjectRepository(db).create("proj", "/repo", "main", "goal", 100.0)


def test_project_lifecycle(db, project_id):
    projects = ProjectRepository(db)
    row = projects.get(project_id)
    assert row["status"] == "CREATED"
    projects.set_status(project_id, ProjectState.PLANNING)
    projects.set_status(project_id, ProjectState.READY)
    assert projects.get(project_id)["status"] == "READY"
    spent = projects.add_spent(project_id, 1.25)
    assert spent == pytest.approx(1.25)


def test_task_ordering_and_insertion(db, project_id):
    tasks = TaskRepository(db)
    tasks.create(project_id, "T001", "first")
    tasks.create(project_id, "T002", "second")
    tasks.create(project_id, "T003", "third")
    # insert between T002 (seq 20) and T003 (seq 30)
    tasks.create(project_id, "T002A", "inserted", sequence=25)
    keys = [t["task_key"] for t in tasks.list_for_project(project_id)]
    assert keys == ["T001", "T002", "T002A", "T003"]


def test_next_runnable_respects_dependencies(db, project_id):
    tasks = TaskRepository(db)
    t1 = tasks.create(project_id, "T001", "first")
    tasks.create(project_id, "T002", "second", dependencies=["T001"])
    assert tasks.next_runnable(project_id)["task_key"] == "T001"
    tasks.set_status(t1, TaskState.READY)
    for state in (TaskState.ANALYZING, TaskState.EXECUTING, TaskState.TESTING,
                  TaskState.VERIFYING, TaskState.REVIEWING, TaskState.COMPLETED):
        tasks.set_status(t1, state)
    assert tasks.next_runnable(project_id)["task_key"] == "T002"


def test_attempts_and_consecutive_failures(db, project_id):
    tasks = TaskRepository(db)
    t1 = tasks.create(project_id, "T001", "first")
    a1 = tasks.start_attempt(t1, "abc123")
    tasks.finish_attempt(a1, AttemptState.FAILED)
    a2 = tasks.start_attempt(t1, "abc123")
    tasks.finish_attempt(a2, AttemptState.FAILED)
    assert tasks.consecutive_failures(t1) == 2
    a3 = tasks.start_attempt(t1, "abc123")
    tasks.finish_attempt(a3, AttemptState.PASSED)
    assert tasks.consecutive_failures(t1) == 0
    assert tasks.get(t1)["attempt_count"] == 3


def test_illegal_task_transition_rejected(db, project_id):
    tasks = TaskRepository(db)
    t1 = tasks.create(project_id, "T001", "first")
    with pytest.raises(Exception):
        tasks.set_status(t1, TaskState.COMPLETED)
    # force bypasses (used by recovery / exceptional paths)
    tasks.set_status(t1, TaskState.COMPLETED, force=True)


def test_runs_and_events(db, project_id):
    tasks = TaskRepository(db)
    runs = RunRepository(db)
    events = EventRepository(db)
    t1 = tasks.create(project_id, "T001", "first")
    a1 = tasks.start_attempt(t1, None)
    r1 = runs.start_run(project_id, "developer", a1)
    runs.finish_run(r1, "COMPLETED", session_id="s1", cost_usd=0.5,
                    token_usage={"input": 100})
    assert runs.count_runs_for_task(t1) == 1
    runs.record_evaluation(a1, "REPAIR", {"verdict": "REPAIR"})
    events.emit("TASK_STARTED", project_id=project_id, task_id=t1)
    rows = events.list_for_project(project_id)
    assert rows[0]["event_type"] == "TASK_STARTED"
