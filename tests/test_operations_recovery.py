"""Operation intent/result journal and crash reconciliation."""

import pytest

from harness.artifacts.manager import ArtifactManager
from harness.database.connection import Database
from harness.database.event_repository import EventRepository
from harness.database.operation_repository import (
    OperationRepository,
    OperationStatus,
    OperationType,
)
from harness.database.project_repository import ProjectRepository
from harness.database.run_repository import RunRepository
from harness.database.task_repository import TaskRepository
from harness.git.checkpoint import OPERATION_TRAILER, CheckpointManager
from harness.git.repository import GitRepository
from harness.orchestrator.recovery import RecoveryManager
from harness.orchestrator.state_machine import AttemptState, TaskState


@pytest.fixture
def world(tmp_path):
    """A wired mini-workspace: git repo + db + repos + recovery."""
    git = GitRepository(tmp_path / "repo")
    git.init()
    (git.path / "hello.txt").write_text("hello\n")
    git.add_all()
    git.commit("initial")

    db = Database(tmp_path / "harness.db")
    events = EventRepository(db)
    projects = ProjectRepository(db, events)
    tasks = TaskRepository(db, events)
    runs = RunRepository(db)
    operations = OperationRepository(db, events)
    artifacts = ArtifactManager(tmp_path / "artifacts")
    checkpoint = CheckpointManager(git, operations)
    recovery = RecoveryManager(
        tasks, events, artifacts, git, checkpoint, operations=operations, runs=runs
    )
    pid = projects.create("p", str(git.path), "main", "g", 10.0)

    class World:
        pass

    w = World()
    w.git, w.db, w.events, w.projects, w.tasks, w.runs = git, db, events, projects, tasks, runs
    w.operations, w.artifacts, w.checkpoint, w.recovery, w.pid = (
        operations, artifacts, checkpoint, recovery, pid
    )
    yield w
    db.close()


def commit_count(git: GitRepository) -> int:
    return len(git._run("rev-list", "HEAD").stdout.split())


def test_commit_records_intent_and_result_with_trailers(world):
    tid = world.tasks.create(world.pid, "T001", "task")
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    (world.git.path / "feature.txt").write_text("x\n")

    commit = world.checkpoint.commit_task(
        "T001", "add feature", project_id=world.pid, task_id=tid, attempt_id=aid
    )

    assert commit
    op = world.db.query_one("SELECT * FROM operations WHERE operation_type = 'GIT_COMMIT'")
    # the RESULT is deliberately left to the caller's completion transaction:
    # until then the intent stays PENDING (reconcilable from the trailer)
    assert op["status"] == "PENDING"
    assert world.checkpoint.pending_operation_id == op["operation_id"]
    message = world.git.commit_message("HEAD")
    assert f"{OPERATION_TRAILER}: {op['operation_id']}" in message
    assert "Harness-Task: T001" in message

    # caller-side finalize (as _complete_task does, atomically with task state)
    world.operations.record_result(
        op["operation_id"], OperationStatus.COMPLETED, {"commit": commit}
    )
    rows = world.db.query_all(
        "SELECT event_type FROM events WHERE operation_id = ? ORDER BY id",
        (op["operation_id"],),
    )
    assert [r["event_type"] for r in rows] == ["GIT_COMMIT_INTENT", "GIT_COMMIT_RESULT"]


def test_reconcile_commit_done_but_db_not_updated(world):
    """Crash window: git commit executed, process died before the DB was
    updated. Recovery must reconcile the DB to git — never commit twice."""
    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    world.tasks.set_status(tid, TaskState.REVIEWING, force=True)

    # simulate _complete_task dying right after the git commit:
    (world.git.path / "feature.txt").write_text("x\n")
    op_id = world.operations.record_intent(
        OperationType.GIT_COMMIT,
        {"task_key": "T001", "task_id": tid, "attempt_id": aid,
         "base_head": world.git.head_commit(), "commit_message": "agent(T001): task"},
        project_id=world.pid, task_id=tid, attempt_id=aid,
    )
    world.git.add_all()
    commit = world.git.commit("agent(T001): task", trailers={OPERATION_TRAILER: op_id})
    # crash: no record_result, no task/attempt updates

    before = commit_count(world.git)
    acted = world.recovery.recover(world.projects.get(world.pid))

    assert acted
    assert commit_count(world.git) == before          # no double commit
    task = world.tasks.get(tid)
    assert task["status"] == "COMPLETED"
    assert task["current_commit"] == commit
    assert world.tasks.get_attempt(aid)["status"] == "PASSED"
    op = world.operations.get(op_id)
    assert op["status"] == OperationStatus.RECONCILED.value


def test_unexecuted_commit_intent_is_not_replayed(world):
    """Intent journaled, crash BEFORE the git commit ran: recovery must not
    invent the commit — the operation is closed FAILED and the normal
    interrupted-attempt path takes over."""
    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    world.tasks.set_status(tid, TaskState.REVIEWING, force=True)
    (world.git.path / "feature.txt").write_text("half-done\n")
    op_id = world.operations.record_intent(
        OperationType.GIT_COMMIT,
        {"task_key": "T001", "task_id": tid, "attempt_id": aid},
        project_id=world.pid, task_id=tid, attempt_id=aid,
    )
    # crash before git commit

    before = commit_count(world.git)
    world.recovery.recover(world.projects.get(world.pid))

    assert commit_count(world.git) == before                      # nothing committed
    assert world.operations.get(op_id)["status"] == "FAILED"
    task = world.tasks.get(tid)
    assert task["status"] == "READY"                              # requeued, not completed
    assert task["current_commit"] is None
    assert world.tasks.get_attempt(aid)["status"] == "INTERRUPTED"
    assert not world.git.is_dirty()                               # tree reset for fresh attempt


def test_interrupted_agent_dispatch_operation_is_closed(world):
    op_id = world.operations.record_intent(
        OperationType.AGENT_DISPATCH, {"role": "developer"}, project_id=world.pid
    )
    world.recovery.recover(world.projects.get(world.pid))
    assert world.operations.get(op_id)["status"] == "INTERRUPTED"
    assert world.operations.unfinished() == []


def test_reconciliation_is_atomic(world, monkeypatch):
    """If any part of the commit reconciliation fails, ALL of it rolls back:
    the operation stays PENDING and the next startup retries from scratch,
    instead of a permanent state/ledger mismatch."""
    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    world.tasks.set_status(tid, TaskState.REVIEWING, force=True)
    (world.git.path / "feature.txt").write_text("x\n")
    op_id = world.operations.record_intent(
        OperationType.GIT_COMMIT,
        {"task_key": "T001", "task_id": tid, "attempt_id": aid},
        project_id=world.pid, task_id=tid, attempt_id=aid,
    )
    world.git.add_all()
    world.git.commit("agent(T001): task", trailers={OPERATION_TRAILER: op_id})

    def failing_record_result(*args, **kwargs):
        raise RuntimeError("dies mid-reconciliation")

    monkeypatch.setattr(world.operations, "record_result", failing_record_result)
    op_row = world.operations.get(op_id)
    with pytest.raises(RuntimeError):
        world.recovery._reconcile_git_commit(world.pid, op_row)

    # everything rolled back together — nothing half-applied
    assert world.tasks.get(tid)["status"] == "REVIEWING"
    assert world.tasks.get(tid)["current_commit"] is None
    assert world.tasks.get_attempt(aid)["status"] == "RUNNING"
    assert world.operations.get(op_id)["status"] == "PENDING"

    monkeypatch.undo()
    world.recovery.recover(world.projects.get(world.pid))  # next startup succeeds
    assert world.tasks.get(tid)["status"] == "COMPLETED"
    assert world.operations.get(op_id)["status"] == "RECONCILED"


def test_dirty_tree_from_interrupted_verification_is_recovered(world):
    """A verification in flight (e.g. the final verification, which runs
    outside any attempt) explains a dirty tree: recovery archives and
    resets instead of refusing to start."""
    op_id = world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND,
        {"commands": ["make test"], "label": "final"},
        project_id=world.pid,
    )
    (world.git.path / "hello.txt").write_text("mutated by verification command\n")

    acted = world.recovery.recover(world.projects.get(world.pid))  # must not raise

    assert acted
    assert not world.git.is_dirty()
    assert world.operations.get(op_id)["status"] == "INTERRUPTED"
    diffs = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1 and "mutated by verification" in diffs[0].read_text()


def test_recovery_crash_before_reset_keeps_verification_evidence(world, monkeypatch):
    """If recovery itself dies after touching the DB but before the tree
    reset, the in-flight verification intent must still be PENDING at the
    next startup — otherwise the dirty tree becomes 'unexplained'."""
    world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND,
        {"commands": ["make test"], "label": "final"},
        project_id=world.pid,
    )
    (world.git.path / "hello.txt").write_text("mutated by verification command\n")

    original_discard = world.checkpoint.discard_working_tree

    def crashing_discard():
        raise RuntimeError("recovery dies mid-reset")

    monkeypatch.setattr(world.checkpoint, "discard_working_tree", crashing_discard)
    with pytest.raises(RuntimeError):
        world.recovery.recover(world.projects.get(world.pid))

    # evidence survived the crashed recovery
    assert len(world.operations.unfinished(OperationType.VERIFICATION_COMMAND)) == 1
    assert world.git.is_dirty()

    monkeypatch.setattr(world.checkpoint, "discard_working_tree", original_discard)
    world.recovery.recover(world.projects.get(world.pid))  # next startup succeeds
    assert not world.git.is_dirty()
    assert world.operations.unfinished() == []


def test_dirty_tree_from_interrupted_dispatch_after_partial_recovery(world):
    """Double-crash: a Developer dispatch dirtied the tree, a first recovery
    closed the attempt but died mid-reset. The still-pending dispatch intent
    must explain the dirty tree on the next startup."""
    tid = world.tasks.create(world.pid, "T001", "task")
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    op_id = world.operations.record_intent(
        OperationType.AGENT_DISPATCH, {"role": "developer"},
        project_id=world.pid, task_id=tid, attempt_id=aid,
    )
    (world.git.path / "hello.txt").write_text("half-done developer edit\n")
    # first recovery pass got as far as closing the attempt, then crashed
    world.tasks.finish_attempt(aid, AttemptState.INTERRUPTED)

    acted = world.recovery.recover(world.projects.get(world.pid))  # must not raise

    assert acted
    assert not world.git.is_dirty()
    assert world.operations.get(op_id)["status"] == "INTERRUPTED"
    diffs = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1 and "half-done developer edit" in diffs[0].read_text()


def test_readonly_dispatch_does_not_explain_user_dirty_tree(world):
    """A pending Planner dispatch (read-only, no attempt) cannot have dirtied
    the tree — user edits made while the harness was stopped must be
    preserved, not archived and reset."""
    from harness.orchestrator.recovery import UnexplainedDirtyWorktree

    world.operations.record_intent(
        OperationType.AGENT_DISPATCH, {"role": "planner"}, project_id=world.pid
    )
    (world.git.path / "hello.txt").write_text("precious user edit while stopped\n")

    with pytest.raises(UnexplainedDirtyWorktree):
        world.recovery.recover(world.projects.get(world.pid))

    assert world.git.is_dirty()  # untouched
    assert (world.git.path / "hello.txt").read_text() == "precious user edit while stopped\n"


def test_recovery_never_settles_another_projects_journal(world):
    """A workspace can hold several projects; recovering project A must not
    destroy project B's pending crash evidence."""
    other_pid = world.projects.create("other", "/elsewhere", "main", "g", 10.0)
    other_commit = world.operations.record_intent(
        OperationType.GIT_COMMIT, {"task_key": "X001"}, project_id=other_pid
    )
    other_dispatch = world.operations.record_intent(
        OperationType.AGENT_DISPATCH, {"role": "developer"}, project_id=other_pid
    )

    world.recovery.recover(world.projects.get(world.pid))

    # the other project's journal is untouched — still PENDING for its own startup
    assert world.operations.get(other_commit)["status"] == "PENDING"
    assert world.operations.get(other_dispatch)["status"] == "PENDING"
