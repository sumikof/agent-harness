"""Harness-level invariant validation."""

import json

import pytest

from harness.artifacts.manager import ArtifactManager
from harness.artifacts.schemas import TaskBrief
from harness.database.connection import Database
from harness.database.event_repository import EventRepository
from harness.database.project_repository import ProjectRepository
from harness.database.run_repository import VERIFY_PASS, RunRepository
from harness.database.task_repository import TaskRepository
from harness.git.repository import GitRepository
from harness.orchestrator.invariants import InvariantChecker, Severity
from harness.orchestrator.state_machine import AttemptState, TaskState


@pytest.fixture
def world(tmp_path):
    git = GitRepository(tmp_path / "repo")
    git.init()
    (git.path / "a.txt").write_text("a\n")
    git.add_all()
    git.commit("initial")
    db = Database(tmp_path / "harness.db")
    events = EventRepository(db)
    projects = ProjectRepository(db, events)
    tasks = TaskRepository(db, events)
    runs = RunRepository(db)
    pid = projects.create("p", str(git.path), "main", "g", 10.0)

    class World:
        pass

    w = World()
    w.git, w.db, w.tasks, w.runs, w.pid = git, db, tasks, runs, pid
    w.checker = InvariantChecker(db, git)
    yield w
    db.close()


def codes(violations, severity=None):
    return [v.code for v in violations if severity is None or v.severity == severity]


def make_completed_task(w, key="T001", *, verify=True, review=True, commit=True):
    tid = w.tasks.create(w.pid, key, "task")
    aid = w.tasks.start_attempt(tid, w.git.head_commit())
    if verify:
        w.runs.record_evaluation(aid, VERIFY_PASS, {"passed": True})
    if review:
        w.runs.record_evaluation(aid, "PASS", {"verdict": "PASS"})
    w.tasks.finish_attempt(aid, AttemptState.PASSED)
    if commit:
        (w.git.path / f"{key}.txt").write_text("x\n")
        w.git.add_all()
        w.tasks.set_commit(tid, w.git.commit(f"agent({key}): task"))
    w.tasks.set_status(tid, TaskState.COMPLETED, force=True)
    return tid


def test_healthy_completed_task_has_no_violations(world):
    make_completed_task(world)
    assert world.checker.check_all(world.pid) == []


def test_completed_requires_verification_pass(world):
    make_completed_task(world, verify=False)
    violations = world.checker.check_all(world.pid)
    assert "COMPLETED_WITHOUT_VERIFICATION_PASS" in codes(violations)


def test_completed_requires_reviewer_pass(world):
    make_completed_task(world, review=False)
    violations = world.checker.check_all(world.pid)
    assert "COMPLETED_WITHOUT_REVIEW_PASS" in codes(violations)


def test_completed_requires_passed_attempt(world):
    tid = world.tasks.create(world.pid, "T009", "task")
    world.tasks.set_status(tid, TaskState.COMPLETED, force=True)  # no attempt at all
    violations = world.checker.check_all(world.pid)
    assert "COMPLETED_WITHOUT_PASSED_ATTEMPT" in codes(violations, Severity.ERROR)


def test_completed_commit_must_resolve_in_git(world):
    tid = make_completed_task(world)
    world.tasks.set_commit(tid, "deadbeef" * 5)  # forge an unresolvable commit
    violations = world.checker.check_all(world.pid)
    assert "COMPLETED_COMMIT_UNRESOLVABLE" in codes(violations, Severity.ERROR)


def test_running_agent_runs_within_limit_allowed(world):
    """Parallel RUNNING runs are legal up to the configured limit."""
    for _ in range(3):
        world.runs.start_run(world.pid, "developer")
    world.checker.max_parallel_agent_runs = 4
    assert "CONCURRENT_AGENT_RUNS" not in codes(world.checker.check_all(world.pid))


def test_running_agent_runs_over_limit_flagged(world):
    world.checker.max_parallel_agent_runs = 2
    for _ in range(3):
        world.runs.start_run(world.pid, "developer")
    violations = world.checker.check_all(world.pid)
    assert "CONCURRENT_AGENT_RUNS" in codes(violations, Severity.ERROR)


def test_second_mutating_run_per_attempt_rejected_by_db_constraint(world):
    """One worktree, one writer: the partial unique index refuses a second
    RUNNING mutating run for the same attempt. Non-mutating runs (reviewer)
    and mutating runs on OTHER attempts stay legal."""
    import sqlite3

    tid = world.tasks.create(world.pid, "T001", "t")
    aid = world.tasks.start_attempt(tid, None)
    world.runs.start_run(world.pid, "developer", aid, mutating=True)
    world.runs.start_run(world.pid, "reviewer", aid, mutating=False)  # legal
    tid2 = world.tasks.create(world.pid, "T002", "t2")
    aid2 = world.tasks.start_attempt(tid2, None)
    world.runs.start_run(world.pid, "developer", aid2, mutating=True)  # legal
    with pytest.raises(sqlite3.IntegrityError):
        world.runs.start_run(world.pid, "tester", aid, mutating=True)


def test_concurrent_mutating_runs_per_attempt_flagged(world):
    tid = world.tasks.create(world.pid, "T001", "t")
    aid = world.tasks.start_attempt(tid, None)
    world.runs.start_run(world.pid, "developer", aid, mutating=True)
    world.db.execute("DROP INDEX idx_agent_runs_single_mutating_per_attempt")
    world.runs.start_run(world.pid, "tester", aid, mutating=True)
    violations = world.checker.check_all(world.pid)
    assert "CONCURRENT_MUTATING_RUNS_PER_ATTEMPT" in codes(violations, Severity.ERROR)


def test_running_run_requires_manifest_spec_and_intent(world):
    world.runs.start_run(world.pid, "developer")  # bare run: no provenance
    violations = world.checker.check_all(world.pid)
    errors = codes(violations, Severity.ERROR)
    assert "RUN_WITHOUT_MANIFEST" in errors
    assert "RUN_WITHOUT_RESOLVED_SPEC" in errors
    assert "RUN_WITHOUT_DISPATCH_INTENT" in errors


def test_event_stream_gap_is_an_error(world):
    events = EventRepository(world.db)
    events.emit("PLANNING_STARTED", project_id=world.pid)
    events.emit("PLANNING_STARTED", project_id=world.pid)
    world.db.execute("UPDATE events SET seq = 9 WHERE seq = 2")
    violations = world.checker.check_all(world.pid)
    assert "EVENT_SEQ_GAP" in codes(violations, Severity.ERROR)


def test_artifact_provenance_checks(world, tmp_path):
    artifacts = ArtifactManager(tmp_path / "artifacts")
    path = artifacts.task_artifact_path("T001", "task-brief.json")
    rid = world.runs.start_run(world.pid, "analyst")
    artifacts.save_enveloped(
        path, TaskBrief(task="T001"), producer_role="analyst", producer_run_id=rid,
        base_commit=world.git.head_commit(),
    )
    assert world.checker.check_artifact_provenance(path) == []

    # producer run that does not exist
    data = json.loads(path.read_text())
    data["producer"]["agent_run_id"] = 99999
    data["base_commit"] = "deadbeef" * 5
    path.write_text(json.dumps(data))
    violations = world.checker.check_artifact_provenance(path)
    assert "ARTIFACT_PRODUCER_MISSING" in codes(violations)
    assert "ARTIFACT_BASE_COMMIT_UNRESOLVABLE" in codes(violations)
