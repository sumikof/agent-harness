"""Regressions for the concurrency-safety defects found in review of the
parallel-execution change.

Each test pins one invariant that a naive parallel implementation breaks:
recovery scope, integration mutual exclusion under cancellation, admission
control at the AgentRun limit, and the identity of the LLM gate.
"""

import asyncio
import json
from pathlib import Path

import pytest

import harness.orchestrator.agent_invoker as agent_invoker_module
from harness.agents import analyst
from harness.agents.base import AgentResult, BaseAgentRunner, create_runner
from harness.config import HarnessConfig, ProjectConfig, VerificationConfig
from harness.git.repository import GitRepository
from harness.orchestrator.project import ProjectOrchestrator

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
        project=ProjectConfig(name="safety", goal="goal"),
        workspace_dir=str(workspace),
        verification=VerificationConfig(language="none", commands=["true"]),
    )
    cfg.config_path = REPO_ROOT / "config.yaml"
    return cfg


# -- recovery scope ---------------------------------------------------------


def test_recovery_never_touches_worktrees_it_does_not_own(config, tmp_path):
    """A worktree the harness never created — the operator's own checkout —
    must survive recovery with its branch intact. Force-removing it and
    `branch -D`-ing a branch with unmerged commits would drop their only
    reference."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()

    # (a) the operator's worktree, entirely outside the harness root
    user_tree = tmp_path / "user-checkout"
    orchestrator.git.add_worktree(user_tree, "my-feature", "HEAD")
    user_repo = GitRepository(user_tree)
    (user_tree / "important.txt").write_text("unmerged user work\n")
    user_repo.add_all()
    user_commit = user_repo.commit("user work")

    # (b) a worktree inside the harness root but on a foreign branch
    inside_root = config.worktrees_path / "manual-poke"
    inside_root.parent.mkdir(parents=True, exist_ok=True)
    orchestrator.git.add_worktree(inside_root, "manual-branch", "HEAD")

    orchestrator.recovery.recover(project)

    assert user_tree.is_dir(), "recovery removed a worktree it does not own"
    assert orchestrator.git.branch_exists("my-feature")
    assert orchestrator.git.commit_exists(user_commit)
    assert inside_root.is_dir(), "recovery removed a foreign-branch worktree"
    assert orchestrator.git.branch_exists("manual-branch")


def test_recovery_still_removes_harness_owned_orphans(config):
    """The narrowed scope must not stop real orphan cleanup: a harness
    worktree on a harness task branch with no RUNNING attempt is removed."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    handle = orchestrator.worktrees.create("T001", 1, orchestrator.git.head_commit())
    (handle.path / "wip.txt").write_text("orphaned work\n")

    assert orchestrator.recovery.recover(project)

    assert not handle.path.exists()
    assert not orchestrator.git.branch_exists(handle.branch)
    archived = list((config.artifacts_path / "diagnostics").glob("orphan-*.diff"))
    assert archived, "orphan diff was discarded instead of archived"


# -- integration mutual exclusion ------------------------------------------


async def test_cancelled_integration_never_overlaps_the_next_merge(config, monkeypatch):
    """Cancelling the awaiter cannot stop the worker thread already inside
    `git merge`. The lock must not be released until that thread settles,
    or a second integration would mutate the checkout concurrently."""
    orchestrator = ProjectOrchestrator(config)
    orchestrator.ensure_project()
    manager = orchestrator.integration

    state = {"active": 0, "peak": 0}
    original = manager._merge_locked

    def slow_merge(*args, **kwargs):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            import time
            time.sleep(0.3)          # the uninterruptible git subprocess
            return original(*args, **kwargs)
        finally:
            state["active"] -= 1

    monkeypatch.setattr(manager, "_merge_locked", slow_merge)

    def integration(key):
        return orchestrator.integration.integrate(
            task_key=key, task_commit="HEAD", title="t", project_id=1)

    first = asyncio.create_task(integration("T001"))
    await asyncio.sleep(0.1)         # let the worker thread enter the merge
    first.cancel()
    second = asyncio.create_task(integration("T002"))

    with pytest.raises(asyncio.CancelledError):
        await first
    await second

    assert state["peak"] == 1, "two merges overlapped in the integration checkout"


# -- AgentRun admission -----------------------------------------------------


class BlockingRunner(BaseAgentRunner):
    """Holds its run open until released; records observed concurrency."""

    provider_name = "fake"

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.peak = 0

    async def run(self, spec):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered.set()
        await self.release.wait()
        self.active -= 1
        payload = {"task": "T001", "summary": "s"}
        return AgentResult(status="COMPLETED",
                           output_text=f"```json\n{json.dumps(payload)}\n```")


async def test_agent_run_limit_queues_instead_of_failing(config, monkeypatch):
    """Reaching max_parallel_agent_runs is ordinary contention: the second
    dispatch waits for a slot. Raising there would abort a task whose
    attempt and worktree are already live."""
    config.parallelism.max_parallel_tasks = 4
    config.parallelism.max_parallel_agent_runs = 1
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    runner = BlockingRunner()
    monkeypatch.setattr(agent_invoker_module, "create_runner",
                        lambda *_a, **_k: runner)

    ctx = orchestrator.project_context()
    invocations = []
    for key in ("T001", "T002"):
        tid = orchestrator.tasks.create(pid, key, key)
        aid = orchestrator.tasks.start_attempt(tid, orchestrator.git.head_commit())
        invocations.append(asyncio.create_task(orchestrator.invoker.invoke(
            analyst.SPEC, pid, ctx, task_id=tid, attempt_id=aid)))

    await runner.entered.wait()
    await asyncio.sleep(0.05)        # give the second dispatch time to queue
    assert len(orchestrator.runs.running_runs()) == 1
    runner.release.set()
    results = await asyncio.gather(*invocations)

    assert len(results) == 2         # neither dispatch was turned into a failure
    assert runner.peak == 1
    assert orchestrator.runs.running_runs() == []


# -- LLM gate identity ------------------------------------------------------


def test_configured_llm_pool_is_the_gate_requests_use(config):
    """`parallelism.resource_pools.llm` must govern real admission, and the
    scheduler's metrics must observe that same gate — not an idle duplicate."""
    config.parallelism.resource_pools.llm = 4
    config.parallelism.starvation_rounds = 3
    config.inference.concurrency.max_requests = 16
    orchestrator = ProjectOrchestrator(config)

    gate = orchestrator.pools.llm
    assert gate.slots == 4, "the smaller configured ceiling must win"
    assert gate.starvation_rounds == 3
    assert orchestrator.invoker.llm_gate is gate
    assert orchestrator.task_runner.pools.llm is gate

    runner = create_runner("openai-compatible", config.inference,
                           orchestrator.invoker.llm_gate)
    assert runner.gate is gate, "requests bypass the configured pool"


def test_standalone_runner_falls_back_to_the_process_gate(config):
    runner = create_runner("openai-compatible", config.inference)
    assert runner.gate.slots == config.inference.concurrency.max_requests


# -- split-task dependency ordering ----------------------------------------


def test_split_rewires_dependents_to_replacements(config):
    """A SPLIT original becomes SKIPPED, and SKIPPED satisfies dependencies.
    Without rewiring, a dependent joins the runnable frontier ALONGSIDE the
    replacements and a parallel scheduler can start it against stale code."""
    from harness.artifacts.schemas import Diagnosis, PlannedTask

    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    t0 = orchestrator.tasks.create(pid, "T000", "prerequisite")
    t1 = orchestrator.tasks.create(pid, "T001", "big task", dependencies=["T000"])
    orchestrator.tasks.create(pid, "T002", "dependent", dependencies=["T001"])
    orchestrator.tasks.set_status(t0, __import__(
        "harness.orchestrator.state_machine", fromlist=["TaskState"]
    ).TaskState.COMPLETED, force=True)

    diagnosis = Diagnosis(
        recommendation="SPLIT",
        split_tasks=[
            PlannedTask(task_key="T001A", title="part A"),
            PlannedTask(task_key="T001B", title="part B", dependencies=["T001A"]),
        ],
    )
    inserted = orchestrator.task_runner._insert_split_tasks(pid, t1, diagnosis)
    assert inserted == ["T001A", "T001B"]
    orchestrator.tasks.set_status(t1, __import__(
        "harness.orchestrator.state_machine", fromlist=["TaskState"]
    ).TaskState.SKIPPED, force=True)

    deps = json.loads(orchestrator.tasks.get_by_key(pid, "T002")["dependencies"])
    assert deps == ["T001A", "T001B"], "dependent still points at the split original"

    # replacements inherit what the original waited for
    assert "T000" in json.loads(
        orchestrator.tasks.get_by_key(pid, "T001A")["dependencies"])

    runnable = {t["task_key"] for t in orchestrator.tasks.runnable_tasks(pid)}
    assert "T002" not in runnable, "dependent became runnable before its replacements"
    assert "T001A" in runnable


# -- host resource pools ----------------------------------------------------


def test_agent_commands_draw_from_the_host_pools(tmp_path):
    """Agent-triggered builds/tests consume the same CPU/RAM as harness
    verification, so they must be gated by the same bounded pools."""
    from harness.agents.local_tools import LocalToolExecutor, classify_command
    from harness.orchestrator.state_machine import Role as R

    assert classify_command("cd svc && pytest -q") == "test"
    assert classify_command("mvn -q package") == "build"
    assert classify_command("git status") is None

    async def scenario():
        pool = asyncio.Semaphore(1)
        state = {"active": 0, "peak": 0}

        class Tracking:
            async def __aenter__(self):
                await pool.acquire()
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])

            async def __aexit__(self, *exc):
                state["active"] -= 1
                pool.release()

        executor = LocalToolExecutor(
            role=R.DEVELOPER, cwd=tmp_path, repo_root=tmp_path,
            heavy_test_pool=Tracking(), heavy_build_pool=Tracking(),
        )
        await asyncio.gather(*(
            executor.execute("run_command", {"command": "pytest --version || true"})
            for _ in range(4)
        ))
        return state["peak"]

    assert asyncio.run(scenario()) == 1


def test_light_commands_are_not_gated(tmp_path):
    from harness.agents.local_tools import LocalToolExecutor
    from harness.orchestrator.state_machine import Role as R

    class Forbidden:
        async def __aenter__(self):
            raise AssertionError("a light command must not consume a host pool slot")

        async def __aexit__(self, *exc):
            pass

    executor = LocalToolExecutor(
        role=R.DEVELOPER, cwd=tmp_path, repo_root=tmp_path,
        heavy_test_pool=Forbidden(), heavy_build_pool=Forbidden(),
    )
    out = asyncio.run(executor.execute("run_command", {"command": "echo hi"}))
    assert "exit code: 0" in out


# -- verification cancellation ---------------------------------------------


async def test_verification_settles_before_the_worktree_is_removed(config, monkeypatch):
    """Cancelling the task while verification runs must not delete the
    worktree out from under the still-running verifier."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    tid = orchestrator.tasks.create(pid, "T001", "task")
    env = orchestrator.task_runner._create_env("T001", 1)
    state = {"finished": False, "tree_present_at_exit": None}

    def slow_verify(label, on_step=None):
        import time
        time.sleep(0.3)
        state["tree_present_at_exit"] = env.handle.path.is_dir()
        state["finished"] = True
        from harness.artifacts.schemas import VerificationResult
        return VerificationResult(passed=True, steps=[])

    monkeypatch.setattr(env.verifier, "run", slow_verify)

    async def verify():
        from harness.concurrency import run_thread_uninterruptible
        async with orchestrator.task_runner.pools.heavy_test:
            return await run_thread_uninterruptible(
                env.verifier.run, "T001-a1", None, label="verification")

    task = asyncio.create_task(verify())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # cancellation waited for the verifier instead of racing it
    assert state["finished"], "cancellation returned while the verifier was live"
    assert state["tree_present_at_exit"] is True
    orchestrator.worktrees.remove(env.handle)


# -- stale worktree registrations -------------------------------------------


def test_vanished_worktree_registration_is_pruned(config):
    """A worktree directory can disappear while git still registers it.
    Without a prune the branch cannot be deleted and the next
    `worktree add -b` fails on the existing name, stranding the task."""
    import shutil

    orchestrator = ProjectOrchestrator(config)
    orchestrator.ensure_project()
    handle = orchestrator.worktrees.create("T001", 1, orchestrator.git.head_commit())
    branch = handle.branch
    shutil.rmtree(handle.path)                      # directory gone, registration stays

    orchestrator.worktrees.remove_path(handle.path, branch=branch)

    assert not orchestrator.git.branch_exists(branch)
    assert orchestrator.worktrees.registered_paths() == []
    # the task can start a fresh cycle on the same branch name
    again = orchestrator.worktrees.create("T001", 1, orchestrator.git.head_commit())
    assert again.path.is_dir()
    orchestrator.worktrees.remove(again)


# -- partial sampling overrides ---------------------------------------------


def test_role_sampling_override_is_partial(config):
    """A role that overrides one field must keep the customized global
    profile for every other field, not pydantic's class defaults."""
    config.inference.sampling.top_p = 0.8
    config.inference.sampling.temperature = 0.7
    config.inference.role_sampling = {
        "reviewer": type(config.inference.sampling).model_validate({"temperature": 0.2})
    }

    reviewer = config.inference.sampling_for_role("reviewer")
    assert reviewer.temperature == 0.2      # the explicit override
    assert reviewer.top_p == 0.8            # the customized global, not 0.95
    developer = config.inference.sampling_for_role("developer")
    assert developer.temperature == 0.7 and developer.top_p == 0.8


# -- repeated cancellation --------------------------------------------------


async def test_settlement_wait_survives_repeated_cancellation():
    """A second SIGINT during shutdown must not shorten the wait: returning
    early hands the abort path a worker still writing in a worktree."""
    import threading

    from harness.concurrency import run_thread_uninterruptible

    started = threading.Event()
    state = {"done": False}

    def worker():
        started.set()
        import time
        time.sleep(0.4)
        state["done"] = True
        return "finished"

    task = asyncio.create_task(run_thread_uninterruptible(worker, label="probe"))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()          # second cancellation, mid-settlement
    await asyncio.sleep(0.05)
    task.cancel()          # and a third
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state["done"], "helper returned while its worker was still running"


# -- process groups ---------------------------------------------------------


def test_timeout_kills_the_whole_process_group(tmp_path):
    """A timed-out command's descendants must not survive to keep writing
    into the worktree the harness is about to archive or delete."""
    from harness.process import run_command

    marker = tmp_path / "child-was-alive.txt"
    # The shell starts a background descendant, then blocks. Killing only the
    # direct child leaves the descendant to create the marker.
    script = (
        f"(sleep 1.2; echo alive > {marker}) & "
        "sleep 30"
    )
    result = run_command(script, tmp_path, timeout=1)
    assert result.timed_out and result.exit_code == -1

    import time
    time.sleep(2.0)   # past when the descendant would have written
    assert not marker.exists(), "a descendant outlived the killed command"


def test_run_command_returns_output_and_exit_code(tmp_path):
    from harness.process import run_command

    ok = run_command("echo hello", tmp_path, timeout=10)
    assert ok.exit_code == 0 and "hello" in ok.output and not ok.timed_out
    bad = run_command("exit 3", tmp_path, timeout=10)
    assert bad.exit_code == 3


# -- endpoint authentication ------------------------------------------------


def test_api_key_is_sent_and_scopes_the_client_cache():
    from harness.agents.openai_compat import auth_headers, shared_client

    assert auth_headers("sk-secret") == {"Authorization": "Bearer sk-secret"}
    assert auth_headers("not-needed") == {}   # local vLLM without --api-key
    assert auth_headers("") == {}

    authed = shared_client("http://endpoint/v1", 10, "sk-secret")
    assert authed.headers.get("authorization") == "Bearer sk-secret"
    # a different credential must never reuse another's client
    assert shared_client("http://endpoint/v1", 10, "sk-other") is not authed
    assert shared_client("http://endpoint/v1", 10, "sk-secret") is authed


# -- health gate covers the models roles really use -------------------------


def test_health_gate_checks_every_effective_role_model(config):
    """provider.for_role decides the dispatched model; the gate must
    validate those, not just the inference default."""
    from harness.config import RoleProviderConfig
    from harness.main import _uses_local_inference, local_role_models

    config.provider.roles = {
        "reviewer": RoleProviderConfig(model="qwen-reviewer"),
        "planner": RoleProviderConfig(type="claude", model="claude-opus-5"),
    }
    models = local_role_models(config)
    assert "qwen-reviewer" in models                 # the role override
    assert config.provider.model in models           # the shared default
    assert "claude-opus-5" not in models             # not a local provider

    all_cloud = config.model_copy(deep=True)
    all_cloud.provider.type = "claude"
    all_cloud.provider.roles = {}
    assert local_role_models(all_cloud) == []
    assert not _uses_local_inference(all_cloud)


# -- streaming mode ---------------------------------------------------------


async def test_streaming_mode_is_honored_and_reassembles_tool_calls(tmp_path, monkeypatch):
    """`inference.streaming: true` must actually stream — and SSE tool-call
    fragments must reassemble into the same shape the loop expects."""
    import httpx

    import harness.agents.openai_compat as oc
    from harness.agents.openai_compat import LocalOpenAICompatibleAgentRunner
    from harness.agents.profile import ResolvedAgentRunSpec
    from harness.config import InferenceConfig
    from harness.orchestrator.resources import PrefixAffinityGate

    (tmp_path / "hello.txt").write_text("streamed content\n")
    seen_stream_flags = []

    def sse(*chunks: dict) -> bytes:
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
        return (body + "data: [DONE]\n\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_stream_flags.append(body["stream"])
        if len(seen_stream_flags) == 1:
            # one tool call split across three deltas
            return httpx.Response(200, content=sse(
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call-1",
                     "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'th": "hello'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '.txt"}'}}]}}]},
                {"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
            ), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"content": '```json\n{"ok": '}}]},
            {"choices": [{"delta": {"content": 'true}\n```'}}]},
            {"usage": {"prompt_tokens": 9, "completion_tokens": 4}},
        ), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(oc, "shared_client",
                        lambda base_url, t, k=None: httpx.AsyncClient(
                            base_url=base_url, transport=transport))

    inference = InferenceConfig(streaming=True)
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(2))
    spec = ResolvedAgentRunSpec(
        provider="openai-compatible", model="m", role="analyst", profile_id="x",
        profile_version="1", profile_hash="h", max_turns=5,
        cwd=str(tmp_path), repo_root=str(tmp_path), base_url="http://fake/v1",
        system_prompt="s", prompt="p",
    )
    result = await runner.run(spec)

    assert all(seen_stream_flags), "streaming was configured but not requested"
    assert result.status == "COMPLETED"
    assert result.structured_output == {"ok": True}
    assert result.telemetry["tool_calls"] == 1     # the split call reassembled
    assert result.token_usage["output_tokens"] == 7
