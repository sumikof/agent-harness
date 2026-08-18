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
    # A task-branch commit is NOT completion under the integration model:
    # the attempt and commit are settled, and the task requeues for a fresh
    # cycle (completion only ever comes from a serialized integration merge).
    assert task["task_commit"] == commit
    assert task["status"] == "READY"
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
    commit = world.git.head_commit()
    with pytest.raises(RuntimeError):
        world.recovery._reconcile_executed_commit(world.pid, op_row, commit)

    # everything rolled back together — nothing half-applied
    assert world.tasks.get(tid)["status"] == "REVIEWING"
    assert world.tasks.get(tid)["task_commit"] is None
    assert world.tasks.get_attempt(aid)["status"] == "RUNNING"
    assert world.operations.get(op_id)["status"] == "PENDING"

    monkeypatch.undo()
    world.recovery.recover(world.projects.get(world.pid))  # next startup succeeds
    assert world.tasks.get(tid)["task_commit"] is not None
    assert world.tasks.get(tid)["status"] == "READY"   # requeued for a fresh cycle
    assert world.operations.get(op_id)["status"] == "RECONCILED"


def diff_hash(world) -> str:
    return world.git.dirty_state_hash()


def test_dirty_tree_matching_verification_intent_is_recovered(world):
    """A verification in flight (e.g. the final verification, which runs
    outside any attempt) explains a dirty tree when the diff hashes to a
    state the harness recorded for it: recovery archives and resets."""
    (world.git.path / "hello.txt").write_text("uncommitted state before verify\n")
    op_id = world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND,
        {"commands": ["make test"], "label": "final", "base_diff_sha256": diff_hash(world)},
        project_id=world.pid,
    )
    # crash before/while the commands ran; the tree is still the journaled state

    acted = world.recovery.recover(world.projects.get(world.pid))  # must not raise

    assert acted
    assert not world.git.is_dirty()
    assert world.operations.get(op_id)["status"] == "INTERRUPTED"
    diffs = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1 and "before verify" in diffs[0].read_text()


def test_unverifiable_dirty_tree_is_refused_even_with_pending_intent(world):
    """A diff matching NO recorded hash may contain user edits: recovery
    fails safe and refuses, even though an intent is pending."""
    from harness.orchestrator.recovery import UnexplainedDirtyWorktree

    world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND,
        {"commands": ["make test"], "label": "final", "base_diff_sha256": "0" * 64},
        project_id=world.pid,
    )
    (world.git.path / "hello.txt").write_text("could be verification, could be user\n")

    with pytest.raises(UnexplainedDirtyWorktree):
        world.recovery.recover(world.projects.get(world.pid))
    assert world.git.is_dirty()  # preserved


def test_recovery_crash_before_reset_keeps_verification_evidence(world, monkeypatch):
    """If recovery itself dies after touching the DB but before the tree
    reset, the in-flight verification intent must still be PENDING at the
    next startup — otherwise the dirty tree becomes 'unexplained'."""
    (world.git.path / "hello.txt").write_text("mutated by verification command\n")
    world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND,
        {"commands": ["make test"], "label": "final", "base_diff_sha256": diff_hash(world)},
        project_id=world.pid,
    )

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
    # first recovery pass annotated the intent and closed the attempt,
    # then crashed during the reset
    world.recovery._annotate_settlement(
        [world.operations.get(op_id)], diff_hash(world))
    world.tasks.finish_attempt(aid, AttemptState.INTERRUPTED)

    acted = world.recovery.recover(world.projects.get(world.pid))  # must not raise

    assert acted
    assert not world.git.is_dirty()
    assert world.operations.get(op_id)["status"] == "INTERRUPTED"
    diffs = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1 and "half-done developer edit" in diffs[0].read_text()


def test_unexecuted_commit_intent_survives_crashed_recovery(world):
    """Double-crash: commit intent journaled but never executed, a first
    recovery closed the attempt then died mid-reset. The still-PENDING
    intent must explain the dirty tree on the next startup."""
    tid = world.tasks.create(world.pid, "T001", "task")
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    (world.git.path / "feature.txt").write_text("passed but uncommitted work\n")
    op_id = world.operations.record_intent(
        OperationType.GIT_COMMIT,
        {"task_key": "T001", "task_id": tid, "attempt_id": aid,
         "diff_sha256": diff_hash(world)},
        project_id=world.pid, task_id=tid, attempt_id=aid,
    )
    # first recovery pass closed the attempt, then crashed before the reset
    world.tasks.finish_attempt(aid, AttemptState.INTERRUPTED)

    acted = world.recovery.recover(world.projects.get(world.pid))  # must not raise

    assert acted
    assert not world.git.is_dirty()
    assert world.operations.get(op_id)["status"] == "FAILED"  # closed after the reset
    diffs = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1 and "passed but uncommitted" in diffs[0].read_text()


def test_stale_commit_intent_does_not_reset_new_user_edits(world):
    """Triple-crash tail: the intent's reset already completed, the process
    died before closing it, and the user edited the tree while stopped. The
    stale intent's diff hash no longer matches, so the user's work is
    preserved, not archived and reset."""
    from harness.orchestrator.recovery import UnexplainedDirtyWorktree

    tid = world.tasks.create(world.pid, "T001", "task")
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    world.operations.record_intent(
        OperationType.GIT_COMMIT,
        {"task_key": "T001", "task_id": tid, "attempt_id": aid,
         "diff_sha256": "0" * 64},  # hash of the ORIGINAL (already reset) diff
        project_id=world.pid, task_id=tid, attempt_id=aid,
    )
    world.tasks.finish_attempt(aid, AttemptState.INTERRUPTED)  # settled earlier
    # tree is clean (reset completed) — then the user edits while stopped
    (world.git.path / "hello.txt").write_text("brand new user edit\n")

    with pytest.raises(UnexplainedDirtyWorktree):
        world.recovery.recover(world.projects.get(world.pid))

    assert world.git.is_dirty()  # untouched
    assert (world.git.path / "hello.txt").read_text() == "brand new user edit\n"


def test_stale_settled_intent_does_not_reset_new_user_edits(world):
    """A dispatch/verification intent that a previous recovery already
    settled (annotated + reset) must not explain NEW user edits made while
    the process was down before the intent got closed."""
    from harness.orchestrator.recovery import UnexplainedDirtyWorktree

    tid = world.tasks.create(world.pid, "T001", "task")
    aid = world.tasks.start_attempt(tid, world.git.head_commit())
    op_id = world.operations.record_intent(
        OperationType.AGENT_DISPATCH, {"role": "developer"},
        project_id=world.pid, task_id=tid, attempt_id=aid,
    )
    world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND, {"commands": ["make"], "label": "final"},
        project_id=world.pid,
    )
    (world.git.path / "hello.txt").write_text("agent half-done work\n")

    # first recovery pass: annotates + archives + resets, then "crashes"
    # before closing the intents
    original_close = world.recovery._close_interrupted_operations

    def crashing_close(pid):
        raise RuntimeError("dies before closing intents")

    world.recovery._close_interrupted_operations = crashing_close
    with pytest.raises(RuntimeError):
        world.recovery.recover(world.projects.get(world.pid))
    world.recovery._close_interrupted_operations = original_close
    assert not world.git.is_dirty()  # first pass did reset the agent diff
    assert world.operations.get(op_id)["status"] == "PENDING"  # but never closed

    # user edits while the harness is stopped
    (world.git.path / "hello.txt").write_text("brand new user edit\n")

    with pytest.raises(UnexplainedDirtyWorktree):
        world.recovery.recover(world.projects.get(world.pid))
    assert (world.git.path / "hello.txt").read_text() == "brand new user edit\n"


def test_evidence_hashing_leaves_user_index_untouched(world):
    """Deciding that a tree is user work must not leave intent-to-add
    entries behind for the user's untracked files."""
    from harness.orchestrator.recovery import UnexplainedDirtyWorktree

    world.operations.record_intent(
        OperationType.GIT_COMMIT,
        {"task_key": "T001", "diff_sha256": "0" * 64},  # stale, never matches
        project_id=world.pid,
    )
    (world.git.path / "user-notes.txt").write_text("untracked user file\n")

    with pytest.raises(UnexplainedDirtyWorktree):
        world.recovery.recover(world.projects.get(world.pid))

    status = world.git._run("status", "--porcelain").stdout
    assert "?? user-notes.txt" in status  # still untracked, not intent-to-add


def test_verification_side_effects_stay_recoverable_per_step(world, tmp_path):
    """Verification commands that mutate the tree refresh the intent's
    recorded diff hash after every command, so a crash between commands
    (or after the last one, before the result) is still settled by
    recovery instead of refused."""
    from harness.config import VerificationConfig
    from harness.verification.runner import VerificationRunner

    verifier = VerificationRunner(
        VerificationConfig(language="none",
                           commands=["sh -c 'echo generated > generated.txt'"]),
        world.git.path, tmp_path / "logs",
    )
    op_id = world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND,
        {"commands": verifier.commands(), "label": "final",
         "base_diff_sha256": diff_hash(world)},
        project_id=world.pid,
    )
    verifier.run(label="final", on_step=lambda: world.operations.annotate(
        op_id, {"base_diff_sha256": diff_hash(world)}))
    # crash here: command ran (tree mutated), result never recorded

    acted = world.recovery.recover(world.projects.get(world.pid))  # must not raise

    assert acted
    assert not world.git.is_dirty()
    assert not (world.git.path / "generated.txt").exists()
    assert world.operations.get(op_id)["status"] == "INTERRUPTED"


def test_unborn_repo_readonly_diff_handles_untracked_directories(tmp_path):
    """An unborn repository with an untracked directory must survive the
    read-only diff round trip: no exception, index restored."""
    from harness.git.repository import GitRepository

    git = GitRepository(tmp_path / "unborn")
    git.init()
    (git.path / "pkg").mkdir()
    (git.path / "pkg" / "mod.py").write_text("x = 1\n")

    diff = git.dirty_diff_readonly()  # must not raise

    assert "mod.py" in diff
    assert set(git.untracked_paths()) == {"pkg/"}  # back to untracked


def test_archived_diff_preserves_binary_content(world):
    """The archived diff is the only copy once the tree is reset — binary
    files must survive as a re-applicable patch, not a 'Binary files
    differ' notice."""
    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    world.tasks.start_attempt(tid, world.git.head_commit())
    binary_content = bytes(range(256))
    (world.git.path / "asset.bin").write_bytes(binary_content)

    world.recovery.recover(world.projects.get(world.pid))

    assert not (world.git.path / "asset.bin").exists()  # tree was reset
    diffs = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1
    patch = diffs[0].read_text()
    assert "GIT binary patch" in patch          # real content, not a notice
    world.git.apply_patch(patch)                # and it round-trips
    assert (world.git.path / "asset.bin").read_bytes() == binary_content


def test_snapshot_ignores_textconv_and_external_diff(world):
    """Repository-configured diff filters produce presentation-only output;
    the recovery snapshot must bypass them to stay re-applicable."""
    (world.git.path / ".gitattributes").write_text("*.bin diff=hexy\n")
    world.git._run("config", "diff.hexy.textconv", "cat")
    world.git.add_all()
    world.git.commit("configure textconv")
    binary_content = bytes(range(256))
    (world.git.path / "asset.bin").write_bytes(binary_content)

    patch = world.git.snapshot_dirty()

    assert "GIT binary patch" in patch  # not a textconv-rendered text hunk
    world.checkpoint.discard_working_tree()
    assert not (world.git.path / "asset.bin").exists()
    world.git.apply_patch(patch)
    assert (world.git.path / "asset.bin").read_bytes() == binary_content


def test_archived_diff_preserves_non_utf8_text(world):
    """Files git treats as text but that are not UTF-8 (e.g. Latin-1) must
    survive archive → reset → re-apply byte-exactly, not crash decoding."""
    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    world.tasks.start_attempt(tid, world.git.head_commit())
    latin1_content = "café résumé\n".encode("latin-1")
    (world.git.path / "notes.txt").write_bytes(latin1_content)

    world.recovery.recover(world.projects.get(world.pid))

    assert not (world.git.path / "notes.txt").exists()  # tree was reset
    diffs = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(diffs) == 1
    world.git.apply_patch_bytes(diffs[0].read_bytes())
    assert (world.git.path / "notes.txt").read_bytes() == latin1_content


def test_non_utf8_diff_still_matches_evidence_hash(world):
    """Evidence hashing works on raw bytes, so a pending verification that
    produced non-UTF-8 content is still settled instead of refused."""
    (world.git.path / "notes.txt").write_bytes("café résumé\n".encode("latin-1"))
    op_id = world.operations.record_intent(
        OperationType.VERIFICATION_COMMAND,
        {"commands": ["make"], "label": "final", "base_diff_sha256": diff_hash(world)},
        project_id=world.pid,
    )

    acted = world.recovery.recover(world.projects.get(world.pid))  # must not raise

    assert acted
    assert not world.git.is_dirty()
    assert world.operations.get(op_id)["status"] == "INTERRUPTED"


def test_archive_tar_preserves_pre_clean_filter_bytes(world):
    """A clean filter (LFS-style redaction) rewrites what git diff emits;
    the tar archived alongside the patch must hold the real on-disk bytes."""
    import tarfile

    (world.git.path / ".gitattributes").write_text("*.env filter=redact\n")
    world.git._run("config", "filter.redact.clean", "sed s/SECRET/REDACTED/")
    world.git.add_all()
    world.git.commit("configure clean filter")

    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    world.tasks.start_attempt(tid, world.git.head_commit())
    (world.git.path / "app.env").write_text("TOKEN=SECRET\n")

    world.recovery.recover(world.projects.get(world.pid))

    patches = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.diff"))
    assert len(patches) == 1
    assert b"REDACTED" in patches[0].read_bytes()  # the patch went through the filter...
    tars = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.files.tar"))
    assert len(tars) == 1
    with tarfile.open(tars[0]) as tar:
        content = tar.extractfile("app.env").read()
    assert content == b"TOKEN=SECRET\n"          # ...but the tar holds ground truth


def configure_redact_filter(world):
    (world.git.path / ".gitattributes").write_text("*.env filter=redact\n")
    world.git._run("config", "filter.redact.clean", "sed s/SECRET/REDACTED/")
    world.git.add_all()
    world.git.commit("configure clean filter")


def test_state_hash_sees_through_clean_filters(world):
    """Two worktrees whose FILTERED diffs coincide must still hash
    differently when the on-disk bytes differ — otherwise a stale intent
    could misattribute user edits."""
    configure_redact_filter(world)
    (world.git.path / "app.env").write_text("TOKEN=SECRET\n")
    hash_secret = world.git.dirty_state_hash()
    (world.git.path / "app.env").write_text("TOKEN=REDACTED\n")
    hash_redacted = world.git.dirty_state_hash()
    assert hash_secret != hash_redacted


def test_archive_tar_written_even_when_filtered_patch_is_empty(world):
    """A clean filter can normalize the diff to empty while the on-disk
    bytes still differ from HEAD; the real files must be tarred anyway."""
    import tarfile

    configure_redact_filter(world)
    (world.git.path / "app.env").write_text("TOKEN=REDACTED\n")
    world.git.add_all()
    world.git.commit("commit redacted env")
    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    world.tasks.start_attempt(tid, world.git.head_commit())
    # worktree SECRET cleans to REDACTED == index -> patch is empty, tree dirty
    (world.git.path / "app.env").write_text("TOKEN=SECRET\n")
    assert world.git.is_dirty()
    assert not world.git.snapshot_dirty_bytes().strip()

    world.recovery.recover(world.projects.get(world.pid))

    tars = list((world.artifacts.root / "diagnostics").glob("interrupted-worktree*.files.tar"))
    assert len(tars) == 1
    with tarfile.open(tars[0]) as tar:
        assert tar.extractfile("app.env").read() == b"TOKEN=SECRET\n"


def test_snapshot_restores_file_to_directory_transition(world):
    """A session that replaced a tracked file with a same-named directory
    must be restorable: the HEAD file is cleared before extraction."""
    (world.git.path / "foo").write_text("plain file\n")
    world.git.add_all()
    world.git.commit("add foo as a file")
    (world.git.path / "foo").unlink()
    (world.git.path / "foo").mkdir()
    (world.git.path / "foo" / "x").write_text("now a directory\n")

    snap = world.git.snapshot_worktree_state()
    world.git.reset_hard("HEAD")
    assert (world.git.path / "foo").is_file()  # HEAD restored the file form

    world.git.restore_worktree_state(snap)     # must not raise

    assert (world.git.path / "foo").is_dir()
    assert (world.git.path / "foo" / "x").read_text() == "now a directory\n"


def test_state_hash_includes_executable_bit(world):
    """A mode-only change (chmod +x) is a real state difference: evidence
    hashes must not treat it as identical to the recorded state."""
    import stat

    script = world.git.path / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    hash_plain = world.git.dirty_state_hash()
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    assert world.git.dirty_state_hash() != hash_plain


def test_recovery_clears_stale_running_runs_of_other_projects(world):
    """The workspace lock makes every remaining RUNNING row a crash
    leftover; recovery must free the workspace-wide max-1-agent slot even
    when the row belongs to another project."""
    other_pid = world.projects.create("other", "/elsewhere", "main", "g", 10.0)
    stale_run = world.runs.start_run(other_pid, "planner")

    world.recovery.recover(world.projects.get(world.pid))

    assert world.runs.running_runs() == []  # slot freed workspace-wide
    row = world.runs.get(stale_run)
    assert row["status"] == "INTERRUPTED"


def test_state_hash_distinguishes_special_nodes_from_deletion(world):
    """A FIFO/socket at a tracked path is a different state than the file
    being deleted — stale evidence must not get a user's node reset away."""
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo not available on this platform")
    target = world.git.path / "hello.txt"
    target.unlink()
    hash_deleted = world.git.dirty_state_hash()
    os.mkfifo(target)
    hash_fifo = world.git.dirty_state_hash()
    assert hash_deleted != hash_fifo
    # replacing the node or changing its permissions is also a new state
    os.chmod(target, 0o600)
    assert world.git.dirty_state_hash() not in (hash_deleted, hash_fifo)


def test_snapshot_preserves_fifo_nodes(world):
    """A FIFO at session start must survive a snapshot/restore round trip,
    not be recorded as a deletion."""
    import os
    import stat as stat_module

    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo not available on this platform")
    target = world.git.path / "hello.txt"
    target.unlink()
    os.mkfifo(target)

    snap = world.git.snapshot_worktree_state()
    world.git.reset_hard("HEAD")
    assert target.is_file()  # HEAD restored the plain file

    world.git.restore_worktree_state(snap)
    assert stat_module.S_ISFIFO(os.lstat(target).st_mode)


def test_snapshot_refuses_unsupported_special_nodes(world):
    """Sockets cannot be captured faithfully: snapshot creation fails loudly
    instead of silently recording a deletion."""
    import socket

    from harness.git.repository import GitError

    # replace a tracked file with a socket: git reports the file deleted,
    # but the path actually holds a node we cannot capture
    sock_path = world.git.path / "hello.txt"
    sock_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(sock_path))
        with pytest.raises(GitError, match="special node"):
            world.git.snapshot_worktree_state()
    finally:
        server.close()
        sock_path.unlink(missing_ok=True)


def test_recovery_refuses_reset_when_archive_cannot_capture_socket(world):
    """A tree holding a socket cannot be archived faithfully; recovery must
    stop before the reset instead of silently dropping the node."""
    import socket
    import stat as stat_module

    from harness.orchestrator.recovery import RecoveryIntegrityError

    tid = world.tasks.create(world.pid, "T001", "task")
    world.tasks.set_status(tid, TaskState.READY)
    world.tasks.start_attempt(tid, world.git.head_commit())
    sock_path = world.git.path / "hello.txt"
    sock_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(sock_path))
        with pytest.raises(RecoveryIntegrityError, match="archive"):
            world.recovery.recover(world.projects.get(world.pid))
        # nothing was reset: the socket survives for the operator to inspect
        assert stat_module.S_ISSOCK((sock_path).lstat().st_mode)
    finally:
        server.close()
        sock_path.unlink(missing_ok=True)


def test_directory_symlinks_inside_untracked_trees_are_archived(world):
    """A symlink-to-directory inside an untracked tree must be walked,
    hashed, archived, and restored — not dropped by os.walk."""
    import os
    import tarfile

    tree = world.git.path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "data.txt").write_text("payload\n")
    os.symlink("sub", tree / "link")

    assert "tree/link" in world.git.expanded_changed_files()

    snap = world.git.snapshot_worktree_state()
    world.git.reset_hard("HEAD")
    assert not tree.exists()
    world.git.restore_worktree_state(snap)
    assert (tree / "link").is_symlink() and os.readlink(tree / "link") == "sub"

    tar_path = world.artifacts.root / "diagnostics" / "symlink-test.files.tar"
    world.artifacts.archive_worktree_files(tar_path, world.git.path,
                                           world.git.changed_paths())
    with tarfile.open(tar_path) as tar:
        member = tar.getmember("tree/link")
        assert member.issym() and member.linkname == "sub"


def test_begin_run_claim_is_atomic_across_threads(tmp_path):
    """Two threads racing into begin_run must resolve to exactly one owner."""
    import threading

    from harness.workspace_lock import WorkspaceLock, WorkspaceLocked

    lock = WorkspaceLock(tmp_path / "harness.lock")
    barrier = threading.Barrier(2)
    results: list[str] = []

    def worker():
        barrier.wait()
        try:
            lock.begin_run()
            results.append("ok")
        except WorkspaceLocked:
            results.append("locked")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lock.end_run()
    assert sorted(results) == ["locked", "ok"]


def test_readonly_diff_restores_index_for_awkward_paths(world):
    """Untracked names with spaces or non-ASCII characters (quoted in
    porcelain v1 output) must survive the read-only diff round trip as
    untracked — no intent-to-add residue."""
    (world.git.path / "my notes.txt").write_text("user file with spaces\n")
    (world.git.path / "メモ 帳.txt").write_text("non-ascii user file\n")

    before = set(world.git.untracked_paths())
    assert before == {"my notes.txt", "メモ 帳.txt"}

    world.git.dirty_diff_readonly()

    assert set(world.git.untracked_paths()) == before  # still '??', not ' A'
    status = world.git._run("status", "--porcelain").stdout
    assert not any(line[:2].strip() == "A" for line in status.splitlines())


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
