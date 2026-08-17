import pytest

from harness.artifacts.manager import ArtifactManager
from harness.database.connection import Database
from harness.database.event_repository import EventRepository
from harness.database.project_repository import ProjectRepository
from harness.database.task_repository import TaskRepository
from harness.git.checkpoint import CheckpointManager
from harness.git.repository import GitRepository
from harness.orchestrator.recovery import RecoveryManager, UnexplainedDirtyWorktree
from harness.orchestrator.state_machine import TaskState


@pytest.fixture
def repo(tmp_path):
    git = GitRepository(tmp_path / "repo")
    git.init()
    (git.path / "hello.txt").write_text("hello\n")
    git.add_all()
    git.commit("initial")
    return git


def test_checkpoint_commit(repo):
    checkpoint = CheckpointManager(repo)
    (repo.path / "feature.txt").write_text("new\n")
    commit = checkpoint.commit_task("T001", "add feature")
    assert commit
    assert not repo.is_dirty()
    assert "agent(T001): add feature" in repo.log_oneline()


def test_checkpoint_nothing_to_commit(repo):
    checkpoint = CheckpointManager(repo)
    assert checkpoint.commit_task("T001", "noop") is None


def test_discard_working_tree(repo):
    checkpoint = CheckpointManager(repo)
    (repo.path / "hello.txt").write_text("modified\n")
    (repo.path / "junk.txt").write_text("junk\n")
    assert repo.is_dirty()
    checkpoint.discard_working_tree()
    assert not repo.is_dirty()
    assert (repo.path / "hello.txt").read_text() == "hello\n"
    assert not (repo.path / "junk.txt").exists()


def test_recovery_after_crash(tmp_path, repo):
    db = Database(tmp_path / "harness.db")
    projects = ProjectRepository(db)
    tasks = TaskRepository(db)
    events = EventRepository(db)
    artifacts = ArtifactManager(tmp_path / "artifacts")
    checkpoint = CheckpointManager(repo)

    project_id = projects.create("p", str(repo.path), "main", "g", 10.0)
    task_id = tasks.create(project_id, "T001", "task")
    tasks.set_status(task_id, TaskState.READY)
    tasks.set_status(task_id, TaskState.ANALYZING)
    tasks.set_status(task_id, TaskState.EXECUTING)
    tasks.start_attempt(task_id, repo.head_commit())

    # simulate crash mid-episode: dirty tree, RUNNING attempt, EXECUTING task
    (repo.path / "hello.txt").write_text("half-finished change\n")

    recovery = RecoveryManager(tasks, events, artifacts, repo, checkpoint)
    acted = recovery.recover(projects.get(project_id))

    assert acted
    assert not repo.is_dirty()
    assert tasks.get(task_id)["status"] == "READY"
    assert tasks.running_attempts(project_id) == []
    attempt = db.query_one("SELECT * FROM task_attempts WHERE task_id = ?", (task_id,))
    assert attempt["status"] == "INTERRUPTED"
    diffs = list((artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1
    assert "half-finished" in diffs[0].read_text()
    db.close()


def test_recovery_refuses_unexplained_dirty_tree(tmp_path, repo):
    """User work without a RUNNING attempt must never be reset (Codex P1)."""
    db = Database(tmp_path / "harness.db")
    projects = ProjectRepository(db)
    tasks = TaskRepository(db)
    events = EventRepository(db)
    artifacts = ArtifactManager(tmp_path / "artifacts")
    project_id = projects.create("p", str(repo.path), "main", "g", 10.0)

    # dirty tree, but NO running attempt recorded — this is user work
    (repo.path / "hello.txt").write_text("precious uncommitted user work\n")

    recovery = RecoveryManager(tasks, events, artifacts, repo, CheckpointManager(repo))
    with pytest.raises(UnexplainedDirtyWorktree):
        recovery.recover(projects.get(project_id))

    assert repo.is_dirty()  # untouched
    assert (repo.path / "hello.txt").read_text() == "precious uncommitted user work\n"
    db.close()


def test_recovery_noop_when_clean(tmp_path, repo):
    db = Database(tmp_path / "harness.db")
    projects = ProjectRepository(db)
    tasks = TaskRepository(db)
    events = EventRepository(db)
    artifacts = ArtifactManager(tmp_path / "artifacts")
    project_id = projects.create("p", str(repo.path), "main", "g", 10.0)
    recovery = RecoveryManager(tasks, events, artifacts, repo, CheckpointManager(repo))
    assert not recovery.recover(projects.get(project_id))
    db.close()
