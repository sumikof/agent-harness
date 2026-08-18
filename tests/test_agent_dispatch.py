"""Dispatch preconditions, capability gating, retry layering, spill, loop guard."""

import json
from pathlib import Path

import pytest

import harness.orchestrator.agent_invoker as agent_invoker_module
from harness.agents.base import (
    AgentResult,
    BaseAgentRunner,
    FailureKind,
    classify_provider_error,
)
from harness.agents.profile import AgentCapabilities
from harness.artifacts.manager import ArtifactManager
from harness.config import HarnessConfig, ProjectConfig, VerificationConfig
from harness.git.repository import GitRepository
from harness.orchestrator.agent_invoker import AgentConfigurationError, ConcurrentRunError
from harness.orchestrator.project import ProjectOrchestrator
from harness.orchestrator.state_machine import ProjectState, Role
from harness.security.hooks import RepeatActionGuard

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config(tmp_path) -> HarnessConfig:
    workspace = tmp_path / "workspace"
    repo = GitRepository(workspace / "repository")
    repo.init()
    (repo.path / "README.md").write_text("seed\n")
    repo.add_all()
    repo.commit("initial")
    cfg = HarnessConfig(
        project=ProjectConfig(name="dispatch-test", goal="goal"),
        workspace_dir=str(workspace),
        verification=VerificationConfig(language="none", commands=["true"]),
    )
    cfg.config_path = REPO_ROOT / "config.yaml"
    return cfg


class ScriptedRunner(BaseAgentRunner):
    """Yields queued AgentResults; records the specs it ran with."""

    provider_name = "fake"

    def __init__(self, results):
        self.results = list(results)
        self.specs = []

    async def run(self, spec):
        self.specs.append(spec)
        return self.results.pop(0)


def ok_result(payload: dict | None = None) -> AgentResult:
    payload = payload or {"summary": "s", "tasks": [{"task_key": "T001", "title": "t"}]}
    return AgentResult(status="COMPLETED",
                       output_text=f"```json\n{json.dumps(payload)}\n```", cost_usd=0.01)


def install(monkeypatch, runner):
    monkeypatch.setattr(agent_invoker_module, "create_runner", lambda *_a, **_k: runner)
    monkeypatch.setattr(agent_invoker_module, "TECHNICAL_RETRY_DELAY", 0.0)


async def invoke_planner(orchestrator):
    from harness.agents import planner
    project = orchestrator.ensure_project()
    return await orchestrator.invoker.invoke(
        planner.SPEC, project["id"], orchestrator.project_context()
    )


async def test_manifest_and_spec_are_durable_before_dispatch(config, monkeypatch):
    orchestrator = ProjectOrchestrator(config)
    observed = {}

    class InspectingRunner(ScriptedRunner):
        async def run(self, spec):
            # At dispatch time, the RUNNING run row must already carry the
            # manifest reference and the resolved spec, durably committed.
            row = orchestrator.db.query_one(
                "SELECT * FROM agent_runs WHERE status = 'RUNNING'"
            )
            observed["manifest_path"] = row["context_manifest_path"]
            observed["resolved_spec"] = row["resolved_spec"]
            observed["dispatch_op"] = row["dispatch_operation_id"]
            return await super().run(spec)

    runner = InspectingRunner([ok_result()])
    install(monkeypatch, runner)
    await invoke_planner(orchestrator)

    # references are stored workspace-relative so a moved/restored
    # workspace can still reconstruct past inputs
    assert observed["manifest_path"] and not Path(observed["manifest_path"]).is_absolute()
    manifest_file = orchestrator.artifacts.resolve(observed["manifest_path"])
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text())
    assert manifest["role"] == "planner"
    assert manifest["sections"]["project_context"]["sha256"]
    # the system prompt BODY is persisted, not just its hash
    system_section = manifest["sections"]["system_prompt"]
    assert not Path(system_section["artifact"]).is_absolute()
    assert orchestrator.artifacts.resolve(system_section["artifact"]).read_text(encoding="utf-8")
    spec = json.loads(observed["resolved_spec"])
    assert spec["prompt_sha256"] and spec["profile_hash"]
    assert "prompt" not in spec  # bodies live in the manifest artifacts, not the DB
    # the recorded manifest hash matches the persisted file bytes, so an
    # integrity check against the artifact compares like with like
    from harness.artifacts.manager import sha256_text
    assert spec["context_manifest_hash"] == sha256_text(
        manifest_file.read_text(encoding="utf-8"))
    op = orchestrator.operations.get(observed["dispatch_op"])
    assert op is not None and op["operation_type"] == "AGENT_DISPATCH"
    assert op["status"] == "COMPLETED"  # result recorded after the run


async def test_agent_never_starts_if_manifest_cannot_be_saved(config, monkeypatch):
    orchestrator = ProjectOrchestrator(config)
    runner = ScriptedRunner([ok_result()])
    install(monkeypatch, runner)

    def failing_save_text(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(orchestrator.artifacts, "save_text", failing_save_text)
    with pytest.raises(OSError):
        await invoke_planner(orchestrator)

    assert runner.specs == []  # the agent was never dispatched
    assert orchestrator.db.query_one("SELECT 1 FROM agent_runs") is None


async def test_missing_capability_rejected_before_dispatch(config, monkeypatch):
    orchestrator = ProjectOrchestrator(config)

    class NoHooksRunner(ScriptedRunner):
        def capabilities(self):
            return AgentCapabilities(structured_output=True, tool_permissions=True,
                                     pre_tool_hook=False)

    runner = NoHooksRunner([ok_result()])
    install(monkeypatch, runner)

    with pytest.raises(AgentConfigurationError, match="pre_tool_hook"):
        await invoke_planner(orchestrator)
    assert runner.specs == []  # no silent degraded run

    # and through the project loop it is an explicit FAILED, not a retry storm
    state = await orchestrator.run()
    assert state == ProjectState.FAILED


async def test_transient_retry_does_not_burn_task_attempts(config, monkeypatch):
    """A provider blip retries the same dispatch; attempt_count is only for
    reasoning retries."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    tid = orchestrator.tasks.create(pid, "T001", "task")
    aid = orchestrator.tasks.start_attempt(tid, orchestrator.git.head_commit())

    transient = AgentResult(status="FAILED", error="connection reset by peer",
                            failure_kind=FailureKind.TRANSIENT)
    runner = ScriptedRunner([transient, ok_result({"task": "T001", "summary": "s"})])
    install(monkeypatch, runner)

    from harness.agents import analyst
    brief = await orchestrator.invoker.invoke(
        analyst.SPEC, pid, orchestrator.project_context(),
        task_id=tid, attempt_id=aid,
    )
    assert brief.task == "T001"
    assert len(runner.specs) == 2                                  # provider retry happened
    assert orchestrator.tasks.get(tid)["attempt_count"] == 1       # attempts untouched
    runs = orchestrator.runs.list_runs_for_attempt(aid)
    assert [r["status"] for r in runs] == ["FAILED", "COMPLETED"]  # both recorded


async def test_permanent_failure_is_never_provider_retried(config, monkeypatch):
    orchestrator = ProjectOrchestrator(config)
    permanent = AgentResult(status="FAILED", error="401 authentication failed",
                            failure_kind=FailureKind.PERMANENT)
    runner = ScriptedRunner([permanent, ok_result()])
    install(monkeypatch, runner)

    with pytest.raises(agent_invoker_module.AgentRunFailed, match="permanent"):
        await invoke_planner(orchestrator)
    assert len(runner.specs) == 1  # exactly one call, no blind retry


async def test_agent_run_limit_is_enforced(config, monkeypatch):
    """RUNNING AgentRun count may never exceed max_parallel_agent_runs."""
    config.parallelism.max_parallel_agent_runs = 1
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    # a stale RUNNING run (as if another dispatch were in flight)
    orchestrator.runs.start_run(project["id"], "developer")
    runner = ScriptedRunner([ok_result()])
    install(monkeypatch, runner)

    with pytest.raises(ConcurrentRunError):
        await invoke_planner(orchestrator)
    assert runner.specs == []


async def test_dispatch_allowed_below_agent_run_limit(config, monkeypatch):
    """Parallel dispatch is legal now: an existing RUNNING run does not
    block another one while the configured limit has headroom."""
    config.parallelism.max_parallel_agent_runs = 4
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    orchestrator.runs.start_run(project["id"], "developer")
    runner = ScriptedRunner([ok_result()])
    install(monkeypatch, runner)

    await invoke_planner(orchestrator)
    assert len(runner.specs) == 1


async def test_running_agent_in_another_project_counts_toward_limit(config, monkeypatch):
    """The limit is workspace-wide: a RUNNING run belonging to ANOTHER
    project in the same DB consumes a slot too."""
    config.parallelism.max_parallel_agent_runs = 1
    orchestrator = ProjectOrchestrator(config)
    orchestrator.ensure_project()
    other_pid = orchestrator.projects.create("other-project", "/elsewhere", "main", "g", 10.0)
    orchestrator.runs.start_run(other_pid, "developer")
    runner = ScriptedRunner([ok_result()])
    install(monkeypatch, runner)

    with pytest.raises(ConcurrentRunError):
        await invoke_planner(orchestrator)
    assert runner.specs == []


async def test_cap_exceeded_run_is_recorded_consistently(config, monkeypatch):
    """When a COMPLETED provider result blows the per-run cost cap, the run
    row, dispatch-operation result and events must all say FAILED — recorded
    together — with only the raise outside the transaction."""
    config.budget.agent_run_usd = 1.0
    orchestrator = ProjectOrchestrator(config)
    expensive = ok_result()
    expensive.cost_usd = 5.0
    runner = ScriptedRunner([expensive])
    install(monkeypatch, runner)

    with pytest.raises(agent_invoker_module.AgentRunFailed, match="cap"):
        await invoke_planner(orchestrator)

    run = orchestrator.db.query_one("SELECT * FROM agent_runs")
    assert run["status"] == "FAILED" and "cap" in run["error"]
    op = orchestrator.operations.get(run["dispatch_operation_id"])
    assert op["status"] == "FAILED"                       # not a dangling COMPLETED
    assert orchestrator.db.query_one(
        "SELECT 1 FROM events WHERE event_type = 'AGENT_COMPLETED'") is None
    assert orchestrator.db.query_one(
        "SELECT 1 FROM events WHERE event_type = 'AGENT_FAILED'") is not None
    assert orchestrator.projects.get(1)["spent_usd"] == pytest.approx(5.0)


async def test_artifact_envelope_carries_input_manifest_hash(config, monkeypatch):
    """A stored artifact must name the ContextManifest it was produced from,
    so provenance is verifiable even with the artifact alone."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    tid = orchestrator.tasks.create(pid, "T001", "task")
    aid = orchestrator.tasks.start_attempt(tid, orchestrator.git.head_commit())
    runner = ScriptedRunner([ok_result({"task": "T001", "summary": "s"})])
    install(monkeypatch, runner)

    from harness.agents import analyst
    artifact_path = orchestrator.artifacts.task_artifact_path("T001", "task-brief.json")
    await orchestrator.invoker.invoke(
        analyst.SPEC, pid, orchestrator.project_context(),
        task_id=tid, attempt_id=aid, artifact_path=artifact_path,
    )

    envelope = json.loads(artifact_path.read_text())
    run = orchestrator.db.query_one("SELECT * FROM agent_runs ORDER BY id DESC LIMIT 1")
    assert envelope["input_manifest_hash"]
    assert envelope["input_manifest_hash"] == run["context_manifest_hash"]


def test_forked_child_cannot_reenter_workspace_lock(tmp_path):
    """fork() copies the reentrancy registry; the child must not inherit the
    lock — its acquire has to hit the OS primitive and be refused."""
    import os

    from harness.workspace_lock import WorkspaceLock, WorkspaceLocked

    if not hasattr(os, "fork"):
        pytest.skip("fork not available on this platform")

    lock = WorkspaceLock(tmp_path / "harness.lock")
    lock.acquire()
    pid = os.fork()
    if pid == 0:  # child
        try:
            WorkspaceLock(tmp_path / "harness.lock").acquire()
        except WorkspaceLocked:
            os._exit(42)
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    lock.release()
    assert os.waitstatus_to_exitcode(status) == 42


async def test_concurrent_same_process_run_loops_are_rejected(config, monkeypatch):
    """Two overlapping project loops in ONE process must not both run: the
    second one's recovery would reclaim the first one's live agent run."""
    from harness.workspace_lock import WorkspaceLocked

    first = ProjectOrchestrator(config)
    first.lock.begin_run()  # as if first.run() were still in flight
    try:
        second = ProjectOrchestrator(config)
        with pytest.raises(WorkspaceLocked):
            await second.run()
    finally:
        first.lock.end_run()

    # once the first loop finished, a sequential loop is allowed again
    third = ProjectOrchestrator(config)
    third.lock.begin_run()  # no longer rejected
    third.lock.end_run()


def test_workspace_lock_blocks_second_process(tmp_path):
    """A second harness process on the same workspace must refuse to start —
    otherwise its recovery would 'reclaim' the first process's live run."""
    import subprocess
    import sys

    from harness.workspace_lock import WorkspaceLock

    lock_path = tmp_path / "harness.lock"
    lock = WorkspaceLock(lock_path)
    lock.acquire()
    lock.acquire()  # reentrant within the same process

    probe = (
        "import sys\n"
        "from harness.workspace_lock import WorkspaceLock, WorkspaceLocked\n"
        f"lock = WorkspaceLock({str(lock_path)!r})\n"
        "try:\n"
        "    lock.acquire()\n"
        "except WorkspaceLocked:\n"
        "    sys.exit(42)\n"
        "sys.exit(0)\n"
    )
    result = subprocess.run([sys.executable, "-c", probe],
                            cwd=str(Path(__file__).resolve().parent.parent))
    assert result.returncode == 42  # the other process was refused

    lock.release()
    result = subprocess.run([sys.executable, "-c", probe],
                            cwd=str(Path(__file__).resolve().parent.parent))
    assert result.returncode == 0   # released lock is acquirable again


def test_classify_provider_error():
    assert classify_provider_error("429 rate limit exceeded") == FailureKind.TRANSIENT
    assert classify_provider_error("Connection reset by peer") == FailureKind.TRANSIENT
    assert classify_provider_error("401 Unauthorized") == FailureKind.PERMANENT
    assert classify_provider_error("invalid_api_key") == FailureKind.PERMANENT
    assert classify_provider_error("something never seen") == FailureKind.TRANSIENT


def test_repeat_action_guard_warns_then_aborts():
    guard = RepeatActionGuard(warn_after=3, abort_after=5)
    call = ("Bash", {"command": "pytest"})
    for _ in range(4):
        allowed, _reason = guard.observe(*call)
        assert allowed
    assert guard.warnings and not guard.loop_detected
    allowed, reason = guard.observe(*call)
    assert not allowed and "LOOP_DETECTED" in reason
    assert guard.loop_detected


def test_repeat_action_guard_resets_on_different_call_and_exempts():
    guard = RepeatActionGuard(warn_after=2, abort_after=3, exempt_tools=["Read"])
    for index in range(10):  # different args every time -> never trips
        allowed, _ = guard.observe("Bash", {"command": f"echo {index}"})
        assert allowed
    for _ in range(10):      # exempt tool may poll forever
        allowed, _ = guard.observe("Read", {"file_path": "x"})
        assert allowed
    assert not guard.loop_detected


def test_oversized_output_spills_to_artifact(tmp_path):
    artifacts = ArtifactManager(tmp_path / "artifacts")
    big = "line\n" * 20000  # 100k chars
    spilled = artifacts.spill_text_output("big.log", big, threshold=10000)

    assert spilled.truncated
    assert Path(spilled.artifact_path).read_text() == big     # nothing lost
    assert spilled.total_bytes == len(big.encode())
    preview = spilled.render()
    assert len(preview) < len(big)
    assert spilled.artifact_path in preview                    # locator included
    assert spilled.sha256[:16] in preview

    small = artifacts.spill_text_output("small.log", "tiny", threshold=10000)
    assert not small.truncated and small.render() == "tiny"


def test_spill_preview_honors_small_thresholds(tmp_path):
    """A deployment that lowers max_inline_output_chars must get a preview
    that actually fits it — head+tail never exceed the threshold."""
    artifacts = ArtifactManager(tmp_path / "artifacts")
    text = "x" * 2000
    spilled = artifacts.spill_text_output("small-threshold.log", text, threshold=1000)
    assert spilled.truncated
    assert len(spilled.head) + len(spilled.tail) <= 1000
    assert Path(spilled.artifact_path).read_text() == text  # full copy intact

    # threshold=1 -> tail_chars becomes 0; text[-0:] must NOT leak the whole
    # string back into the preview
    tiny = artifacts.spill_text_output("tiny-threshold.log", text, threshold=1)
    assert tiny.truncated
    assert len(tiny.head) + len(tiny.tail) <= 1


async def test_result_recording_is_atomic(config, monkeypatch):
    """finish_run, dispatch-operation result, billing and events commit
    together: a crash mid-recording rolls all of it back, leaving the
    PENDING dispatch for recovery instead of a half-recorded run."""
    orchestrator = ProjectOrchestrator(config)
    runner = ScriptedRunner([ok_result()])
    install(monkeypatch, runner)

    original_record_cost = orchestrator.budget.record_cost

    def crashing_record_cost(*args, **kwargs):
        raise RuntimeError("dies mid result recording")

    monkeypatch.setattr(orchestrator.budget, "record_cost", crashing_record_cost)
    with pytest.raises(RuntimeError):
        await invoke_planner(orchestrator)

    run = orchestrator.db.query_one("SELECT * FROM agent_runs")
    assert run["status"] == "RUNNING"  # result rollback — nothing half-recorded
    op = orchestrator.operations.get(run["dispatch_operation_id"])
    assert op["status"] == "PENDING"   # recovery will settle it
    assert orchestrator.projects.get(1)["spent_usd"] == 0
    monkeypatch.setattr(orchestrator.budget, "record_cost", original_record_cost)


async def test_complete_task_refuses_without_recorded_evidence(config):
    """The commit checkpoint is gated on recorded verification + review
    evidence, not on control flow having 'obviously' passed through them."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    tid = orchestrator.tasks.create(pid, "T001", "task")
    env = orchestrator.task_runner._create_env("T001", 1)
    try:
        aid = orchestrator.tasks.start_attempt(
            tid, env.git.head_commit(),
            worktree_path=str(env.handle.path), branch=env.handle.branch,
        )
        task_row = orchestrator.tasks.get(tid)
        (env.handle.path / "change.txt").write_text("x\n")

        with pytest.raises(RuntimeError, match="verification PASS"):
            await orchestrator.task_runner._complete_and_integrate(
                pid, tid, "T001", aid, task_row, env, None)

        orchestrator.runs.record_evaluation(aid, "VERIFY_PASS", {"passed": True})
        with pytest.raises(RuntimeError, match="Reviewer PASS"):
            await orchestrator.task_runner._complete_and_integrate(
                pid, tid, "T001", aid, task_row, env, None)

        orchestrator.runs.record_evaluation(aid, "PASS", {"verdict": "PASS"})
        outcome, feedback = await orchestrator.task_runner._complete_and_integrate(
            pid, tid, "T001", aid, task_row, env, None)
        assert outcome is not None and outcome.value == "COMPLETED"
        assert feedback is None
        # the GIT_COMMIT + GIT_INTEGRATION operation results were settled in
        # the same transaction as the task updates — nothing is left PENDING
        assert orchestrator.operations.unfinished() == []
        task = orchestrator.tasks.get(tid)
        assert task["task_commit"] and task["integration_commit"]
        # the change actually reached the integration branch
        assert (orchestrator.git.path / "change.txt").exists()
    finally:
        orchestrator.worktrees.remove_path(env.handle.path)


async def test_loop_detected_run_is_never_adopted(config, monkeypatch):
    """A run that tripped the repeat-action guard fails the dispatch even if
    the provider returned a formally valid COMPLETED result."""
    orchestrator = ProjectOrchestrator(config)
    looping = ok_result()
    looping.loop_detected = True
    runner = ScriptedRunner([looping, ok_result()])
    install(monkeypatch, runner)

    with pytest.raises(agent_invoker_module.AgentRunFailed, match="[Ll]oop"):
        await invoke_planner(orchestrator)

    assert len(runner.specs) == 1  # no provider retry of the same context
    run = orchestrator.db.query_one("SELECT * FROM agent_runs")
    assert run["status"] == "FAILED"
    event = orchestrator.db.query_one(
        "SELECT 1 FROM events WHERE event_type = 'LOOP_DETECTED'"
    )
    assert event is not None


async def test_transient_retry_restores_worktree_for_mutating_roles(config, monkeypatch):
    """A Developer session that half-edited files before a transient failure
    must not be redispatched onto the mutated tree: the partial diff is
    archived and the base state restored first."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    tid = orchestrator.tasks.create(pid, "T001", "task")
    aid = orchestrator.tasks.start_attempt(tid, orchestrator.git.head_commit())
    repo_path = config.repository_path
    tree_states = []

    class HalfEditingRunner(ScriptedRunner):
        async def run(self, spec):
            tree_states.append((repo_path / "junk.py").exists())
            if len(self.specs) == 0:
                (repo_path / "junk.py").write_text("partial\n")  # side effect...
                self.specs.append(spec)
                return AgentResult(status="FAILED", error="connection timeout",
                                   failure_kind=FailureKind.TRANSIENT)  # ...then dies
            self.specs.append(spec)
            (repo_path / "feature.txt").write_text("done\n")
            return ok_result({"task": "T001", "summary": "s"})

    runner = HalfEditingRunner([])
    install(monkeypatch, runner)

    from harness.agents import developer
    await orchestrator.invoker.invoke(
        developer.SPEC, pid, orchestrator.project_context(), task_id=tid, attempt_id=aid,
    )

    assert tree_states == [False, False]  # retry started from a clean base
    assert not (repo_path / "junk.py").exists()
    archived = list((orchestrator.artifacts.root / "diagnostics").glob(
        "developer-run*-transient-retry.diff"))
    assert len(archived) == 1 and "junk.py" in archived[0].read_text()


async def test_tester_retry_restores_developers_uncommitted_work(config, monkeypatch):
    """The Tester runs on top of the Developer's uncommitted implementation;
    a transient Tester failure must restore THAT state — resetting to bare
    HEAD would erase the finished Developer work."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    tid = orchestrator.tasks.create(pid, "T001", "task")
    aid = orchestrator.tasks.start_attempt(tid, orchestrator.git.head_commit())
    repo_path = config.repository_path
    # the Developer's finished, uncommitted implementation
    (repo_path / "impl.py").write_text("developer work\n")
    impl_seen_by_retry = []

    class FlakyTester(ScriptedRunner):
        async def run(self, spec):
            if len(self.specs) == 0:
                self.specs.append(spec)
                (repo_path / "tests").mkdir(exist_ok=True)
                (repo_path / "tests" / "half.py").write_text("partial test\n")
                return AgentResult(status="FAILED", error="connection timeout",
                                   failure_kind=FailureKind.TRANSIENT)
            self.specs.append(spec)
            impl_seen_by_retry.append((repo_path / "impl.py").exists())
            return ok_result({"task": "T001", "summary": "s"})

    runner = FlakyTester([])
    install(monkeypatch, runner)

    from harness.agents import tester
    await orchestrator.invoker.invoke(
        tester.SPEC, pid, orchestrator.project_context(), task_id=tid, attempt_id=aid,
    )

    assert impl_seen_by_retry == [True]                      # developer work survived
    assert (repo_path / "impl.py").read_text() == "developer work\n"
    assert not (repo_path / "tests" / "half.py").exists()    # tester's partial edit undone


async def test_tester_retry_restores_pre_clean_filter_bytes(config, monkeypatch):
    """The retry restore is taken from real file bytes: a clean filter must
    not rewrite the Developer's uncommitted content across a retry."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    repo_path = config.repository_path
    (repo_path / ".gitattributes").write_text("*.env filter=redact\n")
    orchestrator.git._run("config", "filter.redact.clean", "sed s/SECRET/REDACTED/")
    orchestrator.git.add_all()
    orchestrator.git.commit("configure clean filter")
    tid = orchestrator.tasks.create(pid, "T001", "task")
    aid = orchestrator.tasks.start_attempt(tid, orchestrator.git.head_commit())
    (repo_path / "app.env").write_text("TOKEN=SECRET\n")  # developer's work

    class FlakyTester(ScriptedRunner):
        async def run(self, spec):
            if len(self.specs) == 0:
                self.specs.append(spec)
                return AgentResult(status="FAILED", error="connection timeout",
                                   failure_kind=FailureKind.TRANSIENT)
            self.specs.append(spec)
            return ok_result({"task": "T001", "summary": "s"})

    install(monkeypatch, FlakyTester([]))
    from harness.agents import tester
    await orchestrator.invoker.invoke(
        tester.SPEC, pid, orchestrator.project_context(), task_id=tid, attempt_id=aid,
    )

    assert (repo_path / "app.env").read_text() == "TOKEN=SECRET\n"  # not REDACTED


async def test_final_verification_is_journaled(config, monkeypatch):
    """The project-level final verification records a VERIFICATION_COMMAND
    intent/result like every other verification, so a crash mid-run is
    detectable as an unfinished operation."""
    from tests.test_orchestrator_loop import FakeRunner, install_fake

    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()
    assert state == ProjectState.COMPLETED

    final_ops = [
        op for op in orchestrator.db.query_all(
            "SELECT * FROM operations WHERE operation_type = 'VERIFICATION_COMMAND'")
        if json.loads(op["payload"]).get("label") == "final"
    ]
    assert len(final_ops) == 1
    assert final_ops[0]["status"] == "COMPLETED"
    assert orchestrator.operations.unfinished() == []


async def test_final_result_and_project_completion_are_atomic(config, monkeypatch):
    """If judging the final verification fails mid-way, the operation must
    stay PENDING and the project in FINAL_VERIFICATION — never a settled
    operation with an unfinished project that re-runs the commands."""
    from harness.orchestrator.state_machine import ProjectState

    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    ctx = orchestrator.project_context()

    original_set_status = orchestrator.projects.set_status

    def crashing_set_status(project_id_, status, **kwargs):
        if status == ProjectState.COMPLETED:
            raise RuntimeError("dies before project completion is durable")
        return original_set_status(project_id_, status, **kwargs)

    monkeypatch.setattr(orchestrator.projects, "set_status", crashing_set_status)
    with pytest.raises(RuntimeError):
        await orchestrator._finalize(pid, ctx)

    # rolled back together: the intent is still open for recovery
    assert len(orchestrator.operations.unfinished()) == 1
    assert orchestrator.projects.get(pid)["status"] == ProjectState.FINAL_VERIFICATION.value

    # restart path: recovery settles the interrupted operation, then the
    # re-run finalize completes cleanly
    monkeypatch.setattr(orchestrator.projects, "set_status", original_set_status)
    orchestrator.recovery.recover(orchestrator.projects.get(pid))
    state = await orchestrator._finalize(pid, ctx)
    assert state == ProjectState.COMPLETED
    assert orchestrator.operations.unfinished() == []


async def test_repair_produces_fresh_agent_run_without_resume(config, monkeypatch):
    """Review REPAIR must start a brand-new Developer AgentRun (fresh
    session, own manifest) — never resume the previous session."""
    from tests.test_orchestrator_loop import FakeRunner, install_fake

    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["REPAIR", "PASS"])
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()
    assert state == ProjectState.COMPLETED

    dev_runs = orchestrator.db.query_all(
        "SELECT * FROM agent_runs WHERE role = 'developer' ORDER BY id"
    )
    assert len(dev_runs) == 2
    manifests = {r["context_manifest_path"] for r in dev_runs}
    assert len(manifests) == 2  # each run has its own frozen context
    for row in dev_runs:
        spec = json.loads(row["resolved_spec"])
        assert spec["resume_session_id"] is None
