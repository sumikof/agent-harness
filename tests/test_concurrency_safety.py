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


# -- termination escalation -------------------------------------------------


def test_sigterm_ignoring_descendant_is_escalated_to_sigkill(tmp_path, monkeypatch):
    """The shell leader exiting is NOT proof the group is gone: a descendant
    that traps SIGTERM must still be SIGKILLed, or it keeps writing into the
    worktree after the command was declared timed out."""
    import time

    import harness.process as process_module
    from harness.process import run_command

    monkeypatch.setattr(process_module, "TERM_GRACE_SECONDS", 0.3)
    marker = tmp_path / "survivor.txt"
    script = (
        f"bash -c 'trap \"\" TERM; for i in $(seq 30); do sleep 0.1; done; "
        f"echo alive > {marker}' & sleep 30"
    )

    result = run_command(script, tmp_path, timeout=1)
    assert result.timed_out

    time.sleep(3.5)     # past when the trapping descendant would have written
    assert not marker.exists(), "a SIGTERM-ignoring descendant outlived the kill"


# -- read confinement -------------------------------------------------------


def test_read_tools_cannot_escape_the_repository(tmp_path):
    """Writes were confined but reads were not: a prompt in an untrusted
    repository could pull host credentials into the model request."""
    from harness.agents.local_tools import LocalToolExecutor, decide_local_tool_use
    from harness.orchestrator.state_machine import Role as R

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inside.txt").write_text("repository content\n")
    secret = tmp_path / "secret.env"
    secret.write_text("API_KEY=leaked\n")

    for tool, args in (
        ("read_file", {"path": str(secret)}),
        ("read_file", {"path": "../secret.env"}),
        ("list_directory", {"path": str(tmp_path)}),
        ("grep", {"pattern": "API_KEY", "path": str(tmp_path)}),
        ("glob", {"pattern": "../*.env"}),
    ):
        allowed, reason = decide_local_tool_use(R.DEVELOPER, tool, args, repo)
        assert not allowed, f"{tool} escaped the repository with {args}"
        assert "repositor" in reason

    # in-repository reads still work
    allowed, _ = decide_local_tool_use(R.DEVELOPER, "read_file",
                                       {"path": "inside.txt"}, repo)
    assert allowed

    # and the executor refuses independently of the policy layer
    executor = LocalToolExecutor(role=R.DEVELOPER, cwd=repo, repo_root=repo)
    out = asyncio.run(executor.execute("read_file", {"path": str(secret)}))
    assert "leaked" not in out
    inside = asyncio.run(executor.execute("read_file", {"path": "inside.txt"}))
    assert "repository content" in inside


# -- per-model capability probing -------------------------------------------


async def test_health_probes_capabilities_on_every_role_model(monkeypatch):
    """A second served model lacking tool calling must fail startup, not the
    first task that happens to use it."""
    import httpx

    from harness.agents.health import verify_endpoint
    from harness.config import InferenceConfig

    probed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "good"}, {"id": "weak"}]})
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.path.endswith("/metrics"):
            return httpx.Response(404)
        body = json.loads(request.content)
        model = body["model"]
        probed.append(model)
        if body.get("tools"):
            if model == "weak":          # served, but cannot tool-call
                return httpx.Response(200, json={
                    "choices": [{"message": {"role": "assistant", "content": "sorry"}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 2}})
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant",
                "tool_calls": [{"id": "c", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"})}}]}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2}})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant",
                                     "content": '```json\n{"status": "ok"}\n```'}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    report = await verify_endpoint(InferenceConfig(), ["good", "weak"])

    assert set(report.models) == {"good", "weak"}
    assert report.models["good"].ok
    assert not report.models["weak"].tool_calling_ok
    assert not report.ok, "startup passed despite a role model that cannot tool-call"
    assert any("weak" in e for e in report.errors)
    assert "weak" in probed, "the second role model was never probed"


# -- configuration knobs must act, or be refused ----------------------------


def test_context_profile_governs_the_input_budget():
    """Selecting a profile has to change behaviour, not just documentation,
    and a configured budget can never exceed the profile's window."""
    from harness.config import InferenceConfig

    performance = InferenceConfig()
    assert performance.effective_input_budget() == performance.input_budget_tokens

    wide = InferenceConfig(context_profile="maximum", input_budget_tokens=200_000)
    assert wide.max_model_len() == 262144
    assert wide.effective_input_budget() == 200_000     # the profile allows it

    # the same budget under the production profile is clamped to what fits
    narrow = InferenceConfig(input_budget_tokens=200_000)
    assert narrow.effective_input_budget() == (
        narrow.max_model_len() - narrow.max_output_tokens)
    assert narrow.effective_input_budget() < 200_000


def test_unsupported_settings_are_refused_not_ignored():
    """Silently ignoring a knob promises behaviour the harness does not
    deliver; these values are refused at load with an explanation."""
    import pydantic

    from harness.config import DatabaseConfig, GitStrategyConfig, InferenceConfig

    with pytest.raises(pydantic.ValidationError, match="context_profile"):
        InferenceConfig(context_profile="does-not-exist")
    with pytest.raises(pydantic.ValidationError, match="task_worktrees"):
        GitStrategyConfig(task_worktrees=False)
    with pytest.raises(pydantic.ValidationError, match="integration_strategy"):
        GitStrategyConfig(integration_strategy="parallel")
    with pytest.raises(pydantic.ValidationError, match="single_writer"):
        DatabaseConfig(single_writer=False)

    # the supported values still load
    assert GitStrategyConfig().task_worktrees
    assert DatabaseConfig().single_writer
    assert InferenceConfig(context_profile="long").max_model_len() == 131072


def test_orchestrator_budget_follows_the_profile(config):
    from harness.orchestrator.project import ProjectOrchestrator

    config.inference.context_profile = "performance"
    config.inference.input_budget_tokens = 200_000     # cannot fit the window
    orchestrator = ProjectOrchestrator(config)
    # The INITIAL prompt budget sits BELOW the effective input budget: the
    # difference is the reserve one tool turn needs, so the loop's first
    # tool result never forces the unseen newest turn out of the window.
    assert orchestrator.context_builder.input_budget_tokens == (
        config.inference.prompt_budget_tokens())
    assert orchestrator.context_builder.input_budget_tokens < (
        config.inference.effective_input_budget())
    assert orchestrator.context_builder.input_budget_tokens < 200_000
    assert orchestrator.context_builder.input_budget_tokens > 0


# -- benchmark profile selection -------------------------------------------


def _bench_run(label, aggregate, errors=0, spec=None):
    return {
        "label": label,
        "model": "qwen3.6-27b-fp8",
        "server_profile": {
            "SPECULATIVE_CONFIG": json.dumps(spec) if spec else "",
            "MAX_NUM_SEQS": "16",
        },
        "summaries": [{
            "concurrency": 16,
            "aggregate_output_tokens_per_s": aggregate,
            "errors": errors,
            "error_rate": round(errors / 16, 4),
        }],
    }


def test_tuning_profile_picks_the_best_successful_run(tmp_path):
    """Sweep points restart the server, so the LAST invocation is not the
    best one. The frozen profile must be the highest-throughput run that
    completed without errors — carrying that run's own serving config."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import benchmark_inference as bench

    class Args:
        model = "qwen3.6-27b-fp8"
        label = "mtp4"          # the most recent point, deliberately not best

    runs = [
        _bench_run("baseline", 900),
        _bench_run("mtp2", 2100, spec={"method": "qwen3_next_mtp",
                                       "num_speculative_tokens": 2}),
        _bench_run("mtp3", 2450, spec={"method": "qwen3_next_mtp",
                                       "num_speculative_tokens": 3}),
        # fastest, but it dropped requests — never "known-good"
        _bench_run("mtp4", 2600, errors=3, spec={"method": "qwen3_next_mtp",
                                                 "num_speculative_tokens": 4}),
    ]
    bench.write_tuning_profile(tmp_path, Args, runs)

    profile = json.loads((tmp_path / "inference-tuning.json").read_text())
    assert profile["benchmark_label"] == "mtp3"
    assert profile["primary_score"]["value"] == 2450
    assert profile["primary_score"]["error_rate"] == 0.0
    # the winner's OWN serving configuration, not the current .env
    assert profile["profile"]["speculative"]["num_speculative_tokens"] == 3


def test_tuning_profile_refuses_when_no_clean_run_exists(tmp_path):
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import benchmark_inference as bench

    class Args:
        model = "qwen3.6-27b-fp8"
        label = "only"

    bench.write_tuning_profile(tmp_path, Args, [_bench_run("only", 500, errors=2)])
    assert not (tmp_path / "inference-tuning.json").exists()


def test_metric_sum_distinguishes_absent_from_zero(tmp_path):
    """`healthcheck.sh` treats an empty result as 'metric absent'. An awk
    accumulator that prints 0 regardless makes that failure unreachable and
    lets a run be labeled speculative with no speculative metrics."""
    import subprocess

    script = (REPO_ROOT / "deploy" / "inference" / "healthcheck.sh").read_text()
    start = script.index("curl -fsS \"${BASE}/metrics\"")
    awk_program = script[script.index("awk -v pats=", start):]
    awk_program = awk_program[awk_program.index("'") + 1:]
    awk_program = awk_program[:awk_program.index("'")]

    present = tmp_path / "present.txt"
    present.write_text("# HELP x\nvllm:prefix_cache_hits_total 5\n"
                       "vllm:prefix_cache_queries_total 7\n")
    absent = tmp_path / "absent.txt"
    absent.write_text("# HELP x\nvllm:num_requests_running 3\n")

    def run_awk(pattern, path):
        return subprocess.run(["awk", "-v", f"pats={pattern}", awk_program, str(path)],
                              capture_output=True, text=True).stdout

    assert run_awk("prefix_cache", present) == "12"
    assert run_awk("spec_decod", absent) == "", "absent metrics reported as zero"


# -- .env shell semantics ---------------------------------------------------


def test_shipped_env_example_sources_to_valid_speculative_json():
    """start.sh SOURCES .env as Bash, so an unquoted JSON value loses its
    double quotes and vLLM rejects the documented default deployment."""
    import subprocess

    env_example = REPO_ROOT / "deploy" / "inference" / ".env.example"
    value = subprocess.run(
        ["bash", "-c", 'set -a; . "$1"; set +a; printf "%s" "$SPECULATIVE_CONFIG"',
         "bash", str(env_example)],
        capture_output=True, text=True,
    ).stdout
    parsed = json.loads(value)          # raises if the quoting is wrong
    assert parsed["method"] == "qwen3_next_mtp"


def test_env_profile_is_read_with_shell_semantics(tmp_path):
    """A raw '=' split would keep the surrounding quotes, so the recorded
    speculative setting would no longer parse as JSON."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import benchmark_inference as bench

    env_file = tmp_path / ".env"
    env_file.write_text(
        "MAX_NUM_SEQS=16\n"
        "SPECULATIVE_CONFIG='{\"method\":\"qwen3_next_mtp\","
        "\"num_speculative_tokens\":2}'\n"
    )
    profile = bench.read_env_profile(env_file)
    assert profile["MAX_NUM_SEQS"] == "16"
    assert json.loads(profile["SPECULATIVE_CONFIG"])["num_speculative_tokens"] == 2


def test_legacy_runs_without_serving_config_cannot_win(tmp_path):
    """A record written before serving settings were captured would make the
    profile describe hard-coded defaults instead of the winning run."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import benchmark_inference as bench

    class Args:
        model = "qwen3.6-27b-fp8"
        label = "new"

    fastest_legacy = _bench_run("legacy", 9999)
    fastest_legacy["server_profile"] = None
    eligible = _bench_run("mtp2", 2100, spec={"method": "qwen3_next_mtp",
                                              "num_speculative_tokens": 2})
    bench.write_tuning_profile(tmp_path, Args, [fastest_legacy, eligible])

    profile = json.loads((tmp_path / "inference-tuning.json").read_text())
    assert profile["benchmark_label"] == "mtp2"
    assert profile["profile"]["speculative"]["num_speculative_tokens"] == 2


# -- prefix cache verification measures HITS --------------------------------


def test_prefix_cache_check_ignores_query_counters(tmp_path):
    """prefix_cache_queries_total advances for the probe requests whether or
    not anything was cached; summing it would verify an inert cache."""
    import subprocess

    script = (REPO_ROOT / "deploy" / "inference" / "healthcheck.sh").read_text()
    assert "metric_sum prefix_cache hit" in script, "the check no longer targets hits"
    awk_program = script[script.index("awk -v pats=", script.index('${BASE}/metrics')):]
    awk_program = awk_program[awk_program.index("'") + 1:]
    awk_program = awk_program[:awk_program.index("'")]

    queries_only = tmp_path / "q.txt"
    queries_only.write_text("vllm:prefix_cache_queries_total 12\n")
    with_hits = tmp_path / "h.txt"
    with_hits.write_text("vllm:prefix_cache_queries_total 12\n"
                         "vllm:prefix_cache_hits_total 4\n")

    def run_awk(path):
        return subprocess.run(["awk", "-v", "pats=prefix_cache hit",
                               awk_program, str(path)],
                              capture_output=True, text=True).stdout

    assert run_awk(queries_only) == "", "a queries-only endpoint looked verified"
    assert run_awk(with_hits) == "4"


async def test_health_prefix_probe_counts_hits_only(monkeypatch):
    import httpx

    from harness.agents.health import _prefix_cache_counters

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=(
            "vllm:prefix_cache_queries_total 100\n"
            "vllm:prefix_cache_hits_total 7\n"
        ))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await _prefix_cache_counters(client, "http://server") == 7.0


# -- context window arithmetic ----------------------------------------------


def test_output_reservation_cannot_consume_the_window():
    """Clamping to a one-token budget would hide the misconfiguration: every
    request would still reserve more than the server context holds."""
    import pydantic

    from harness.config import InferenceConfig

    with pytest.raises(pydantic.ValidationError, match="max_output_tokens"):
        InferenceConfig(max_output_tokens=65536)          # the whole window
    with pytest.raises(pydantic.ValidationError, match="max_output_tokens"):
        InferenceConfig(max_output_tokens=65024)          # leaves 512

    fits = InferenceConfig(max_output_tokens=8192)
    assert fits.effective_input_budget() > 0
    assert (fits.effective_input_budget() + fits.max_output_tokens
            <= fits.max_model_len())


def test_duplicate_and_unknown_inference_keys_are_refused():
    """`inference.provider` duplicated the authoritative provider.type and
    was never read; unknown keys must fail rather than be ignored."""
    import pydantic

    from harness.config import InferenceConfig

    with pytest.raises(pydantic.ValidationError, match="extra_forbidden"):
        InferenceConfig(provider="openai-compatible")
    with pytest.raises(pydantic.ValidationError, match="extra_forbidden"):
        InferenceConfig(typo_key=1)


# -- read-only roles need a stable tree -------------------------------------

def test_diagnostician_reads_a_snapshot_not_the_moving_integration_branch(config):
    """The integration checkout advances while other tasks merge into it, so
    a read-only role analysing it could read files from different commits —
    or from a merge in progress. It gets a detached snapshot instead."""
    orchestrator = ProjectOrchestrator(config)
    orchestrator.ensure_project()
    pinned = orchestrator.git.head_commit()

    snapshot = orchestrator.worktrees.create_snapshot("T001-diagnosis1", pinned)
    try:
        assert snapshot.path.is_dir()
        assert snapshot.branch is None                       # nothing to commit to
        assert snapshot.repo.head_commit() == pinned
        assert snapshot.repo.current_branch() == "DETACHED"

        # the integration branch moves on; the snapshot does not
        (orchestrator.git.path / "later.txt").write_text("integrated later\n")
        orchestrator.git.add_all()
        moved = orchestrator.git.commit("another task integrated")
        assert orchestrator.git.head_commit() == moved
        assert snapshot.repo.head_commit() == pinned
        assert not (snapshot.path / "later.txt").exists()

        # recovery recognises a crashed snapshot as harness-owned
        assert orchestrator.worktrees.owns(snapshot.path)
        assert orchestrator.worktrees.owns_checkout("DETACHED")
        assert not orchestrator.worktrees.owns_checkout("someones-feature")
    finally:
        orchestrator.worktrees.remove(snapshot)
    assert not snapshot.path.exists()


def test_stale_snapshot_is_reclaimed_not_silently_bypassed(config):
    """A leftover snapshot from a crashed diagnosis must be reclaimed. Failing
    would fall back to reading the moving integration branch — losing exactly
    the isolation the snapshot provides."""
    from harness.git.repository import GitRepository

    orchestrator = ProjectOrchestrator(config)
    orchestrator.ensure_project()
    head = orchestrator.git.head_commit()

    first = orchestrator.worktrees.create_snapshot("T001-diagnosis1", head)
    assert first.path.is_dir()
    # crash: the snapshot is never disposed, and the same label comes round again
    second = orchestrator.worktrees.create_snapshot("T001-diagnosis1", head)
    assert second.path.is_dir()
    assert second.repo.current_branch() == "DETACHED"
    assert second.repo.head_commit() == head
    orchestrator.worktrees.remove(second)

    # a foreign checkout under our root is refused, never clobbered
    foreign = config.worktrees_path / "T001-diagnosis1"
    orchestrator.git.add_worktree(foreign, "someones-branch", head)
    with pytest.raises(Exception, match="refusing to reuse"):
        orchestrator.worktrees.create_snapshot("T001-diagnosis1", head)
    assert GitRepository(foreign).current_branch() == "someones-branch"


def test_deployment_artifacts_are_not_committable():
    """start.sh generates a launch script and the benchmark accumulates raw
    measurements; neither belongs in the repository (the FROZEN tuning
    profile does).

    Asked of git itself rather than matched against .gitignore text — a
    filename mentioned in a comment there is not an ignore rule.
    """
    import subprocess

    def is_ignored(path: str) -> bool:
        return subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=REPO_ROOT, capture_output=True,
        ).returncode == 0

    assert is_ignored("deploy/inference/.generated/launch.sh")
    assert is_ignored("config/benchmark-results.json")
    assert is_ignored("deploy/inference/.env")        # operator's HF token
    # the frozen production profile is source, not an artifact
    assert not is_ignored("config/inference-tuning.json")


# -- tool-loop growth vs the context window ---------------------------------


async def test_tool_loop_history_is_clamped_to_the_input_budget(tmp_path, monkeypatch):
    """The budget shapes the FIRST prompt only; each turn appends an
    assistant message plus a tool result and the whole history is resent.
    One large read_file is enough to overrun the window otherwise."""
    import httpx

    import harness.agents.openai_compat as oc
    from harness.agents.openai_compat import (
        CHARS_PER_TOKEN,
        ELIDED_TOOL_RESULT,
        LocalOpenAICompatibleAgentRunner,
        _messages_size,
    )
    from harness.agents.profile import ResolvedAgentRunSpec
    from harness.config import InferenceConfig
    from harness.orchestrator.resources import PrefixAffinityGate

    big = tmp_path / "big.txt"
    big.write_text("x" * 120_000 + "\n")          # far past one tool result cap
    sizes: list[int] = []

    unelided_latest: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sizes.append(_messages_size(body["messages"]))
        tool_messages = [m for m in body["messages"] if m.get("role") == "tool"]
        if tool_messages:
            unelided_latest.append(
                tool_messages[-1]["content"] != ELIDED_TOOL_RESULT)
        if len(sizes) <= 3:
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"c{len(sizes)}", "type": "function",
                                "function": {"name": "read_file",
                                             "arguments": json.dumps({"path": "big.txt"})}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": '```json\n{"task": "T001", "summary": "s"}\n```'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(oc, "shared_client",
                        lambda base_url, t, k=None: httpx.AsyncClient(
                            base_url=base_url, transport=transport))

    inference = InferenceConfig(input_budget_tokens=4000)   # small, so the loop bites
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(2))
    spec = ResolvedAgentRunSpec(
        provider="openai-compatible", model="m", role="analyst", profile_id="x",
        profile_version="1", profile_hash="h", max_turns=8,
        cwd=str(tmp_path), repo_root=str(tmp_path), base_url="http://fake/v1",
        system_prompt="system", prompt="task prompt",
    )
    result = await runner.run(spec)

    assert result.status == "COMPLETED"
    from harness.agents.local_tools import MAX_TOOL_OUTPUT_CHARS
    limit = inference.effective_input_budget() * CHARS_PER_TOKEN
    assert sizes, "no request was sent"
    # The newest tool result is deliberately NOT elidable (the model has
    # not seen it yet), so the hard bound is budget + one full turn; the
    # elidable part of the history must stay inside the budget itself.
    turn_allowance = MAX_TOOL_OUTPUT_CHARS + 4000
    assert max(sizes) <= limit + turn_allowance, (
        f"history grew to {max(sizes)} chars, past the {limit}-char budget "
        f"plus one unseen turn ({turn_allowance})")
    assert len(sizes) > 3, "the tool loop did not actually run"
    assert unelided_latest and all(unelided_latest), (
        "a request elided the tool result the model had never seen")


def test_elision_keeps_the_assignment_and_tool_call_pairing(tmp_path):
    """Trimming must not break the protocol: a tool message whose assistant
    tool_calls remain cannot simply be dropped, and the assignment stays."""
    from harness.agents.openai_compat import (
        ELIDED_TOOL_RESULT,
        LocalOpenAICompatibleAgentRunner,
    )
    from harness.config import InferenceConfig
    from harness.orchestrator.resources import PrefixAffinityGate

    runner = LocalOpenAICompatibleAgentRunner(
        InferenceConfig(input_budget_tokens=1000), PrefixAffinityGate(2))
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "THE ASSIGNMENT"},
    ]
    for i in range(6):
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"c{i}", "type": "function",
                                         "function": {"name": "read_file",
                                                      "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}",
                         "content": "y" * 5000})

    runner._fit_to_window(messages)

    assert messages[0]["content"] == "SYSTEM PROMPT"
    assert messages[1]["content"] == "THE ASSIGNMENT"
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 6                     # none dropped
    assert all(m.get("tool_call_id") for m in tool_messages)
    assert any(m["content"] == ELIDED_TOOL_RESULT for m in tool_messages)


def test_health_gate_requires_usage_reporting():
    """The runner declares a usage_reporting capability and records absent
    counts as zero, so an endpoint without usage must not pass startup."""
    from harness.agents.health import ModelReport

    complete = ModelReport(model="m", available=True, completion_ok=True,
                           usage_reported=True, tool_calling_ok=True,
                           structured_output_ok=True)
    assert complete.ok
    no_usage = ModelReport(model="m", available=True, completion_ok=True,
                           usage_reported=False, tool_calling_ok=True,
                           structured_output_ok=True)
    assert not no_usage.ok


def test_non_positive_output_reservation_is_refused():
    """Zero or negative leaves MORE apparent headroom, so a bare headroom
    check accepts it — then every request is rejected by the server."""
    import pydantic

    from harness.config import InferenceConfig

    for value in (0, -1):
        with pytest.raises(pydantic.ValidationError):
            InferenceConfig(max_output_tokens=value)
    with pytest.raises(pydantic.ValidationError):
        InferenceConfig(input_budget_tokens=0)
    assert InferenceConfig(max_output_tokens=1).effective_input_budget() > 0


def test_clamping_elides_tool_call_arguments_not_just_results(tmp_path):
    """write_file/edit_file carry the whole file body in the CALL arguments.
    Eliding only the result leaves the larger half of the turn in history."""
    from harness.agents.openai_compat import (
        ELIDED_ARGUMENTS,
        ELIDED_TOOL_RESULT,
        CHARS_PER_TOKEN,
        LocalOpenAICompatibleAgentRunner,
        _messages_size,
    )
    from harness.config import InferenceConfig
    from harness.orchestrator.resources import PrefixAffinityGate

    inference = InferenceConfig(input_budget_tokens=2000)
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(2))
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "ASSIGNMENT"},
    ]
    for i in range(4):
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"w{i}", "type": "function", "function": {
                "name": "write_file",
                # the file body lives HERE, not in the result
                "arguments": json.dumps({"path": f"f{i}.py", "content": "z" * 8000})}}],
        })
        messages.append({"role": "tool", "tool_call_id": f"w{i}",
                         "content": "wrote 8000 chars"})

    runner._fit_to_window(messages)

    limit = inference.effective_input_budget() * CHARS_PER_TOKEN
    # The final turn is the model's unseen result — protected. Everything
    # BEFORE it must have been squeezed inside the budget.
    seen = messages[:-2]
    assert _messages_size(seen) <= limit, "seen history still exceeds the budget"
    assert messages[-1]["content"] != ELIDED_TOOL_RESULT, (
        "the unseen newest tool result was elided")
    assert messages[-2]["tool_calls"][0]["function"]["arguments"] != ELIDED_ARGUMENTS
    calls = [c for m in messages for c in (m.get("tool_calls") or [])]
    assert any(c["function"]["arguments"] == ELIDED_ARGUMENTS for c in calls)
    # protocol pairing survives: every call keeps its id and name, and the
    # compacted arguments are still valid JSON
    for call in calls:
        assert call["id"] and call["function"]["name"]
        json.loads(call["function"]["arguments"])
    assert messages[0]["content"] == "SYSTEM"
    assert messages[1]["content"] == "ASSIGNMENT"


def test_partial_usage_object_does_not_pass_the_health_gate(monkeypatch):
    """`{"completion_tokens": 5}` without prompt_tokens would let every input
    count be recorded as zero."""
    import httpx

    from harness.agents.health import verify_endpoint
    from harness.config import InferenceConfig

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "partial"}]})
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.path.endswith("/metrics"):
            return httpx.Response(404)
        body = json.loads(request.content)
        if body.get("tools"):
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant",
                "tool_calls": [{"id": "c", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"})}}]}}],
                "usage": {"completion_tokens": 5}})          # prompt_tokens missing
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant",
                                     "content": '```json\n{"status": "ok"}\n```'}}],
            "usage": {"completion_tokens": 5}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)

    async def run():
        return await verify_endpoint(InferenceConfig(), ["partial"])

    report = asyncio.run(run())
    assert not report.models["partial"].usage_reported
    assert not report.ok
    assert any("prompt_tokens" in e for e in report.errors)


# -- the health gate must probe the path production actually uses -----------


def _sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode("utf-8")


def _streaming_only_broken_transport():
    """A server that reports usage in JSON completions but omits it from SSE.

    This is exactly the configuration a JSON-only health probe passes and
    a streaming agent loop then mis-records: every turn books zero tokens.
    """
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.path.endswith("/metrics"):
            return httpx.Response(404)
        body = json.loads(request.content)
        tool_call = {"id": "c", "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps({"path": "README.md"})}}
        if body.get("stream"):
            # Same content, same tool call — but no usage anywhere in the stream.
            if body.get("tools"):
                delta = {"role": "assistant", "tool_calls": [dict(tool_call, index=0)]}
            else:
                delta = {"role": "assistant",
                         "content": '```json\n{"status": "ok"}\n```'}
            return httpx.Response(200, content=_sse([{"choices": [{"delta": delta}]}]))
        if body.get("tools"):
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant",
                                         "tool_calls": [tool_call]}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2}})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant",
                                     "content": '```json\n{"status": "ok"}\n```'}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2}})

    return httpx.MockTransport(handler)


def _sse_response(httpx_mod, chunks: list[dict]):
    return httpx_mod.Response(200, content=_sse(chunks))


def _patch_async_client(monkeypatch, transport):
    import httpx

    real_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)


def test_health_gate_probes_the_configured_streaming_path(monkeypatch):
    """With streaming on, the probe must stream — a server that only reports
    usage in JSON must not pass a streaming deployment."""
    from harness.agents.health import verify_endpoint
    from harness.config import InferenceConfig

    _patch_async_client(monkeypatch, _streaming_only_broken_transport())

    streamed = asyncio.run(
        verify_endpoint(InferenceConfig(streaming=True), ["m"]))
    assert not streamed.models["m"].usage_reported, (
        "the health gate passed a streaming deployment whose SSE responses "
        "carry no usage; every streamed turn would record zero tokens")
    assert not streamed.ok
    assert any("prompt_tokens" in e for e in streamed.errors)

    # Capabilities still have to be readable off the stream, otherwise the
    # probe would only be failing because it cannot parse SSE at all.
    assert streamed.models["m"].completion_ok
    assert streamed.models["m"].tool_calling_ok
    assert streamed.models["m"].structured_output_ok


def test_health_gate_still_passes_the_same_server_without_streaming(monkeypatch):
    """The non-streaming path of that same server is genuinely fine — the
    failure above is about the configured path, not a broken endpoint."""
    from harness.agents.health import verify_endpoint
    from harness.config import InferenceConfig

    _patch_async_client(monkeypatch, _streaming_only_broken_transport())

    report = asyncio.run(verify_endpoint(InferenceConfig(streaming=False), ["m"]))
    assert report.models["m"].usage_reported
    assert report.ok


def test_streaming_probe_sends_stream_options_include_usage(monkeypatch):
    """vLLM only emits usage in SSE when asked; a probe that forgets
    stream_options would fail every streaming endpoint."""
    import httpx

    from harness.agents.health import verify_endpoint
    from harness.config import InferenceConfig

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.path.endswith("/metrics"):
            return httpx.Response(404)
        body = json.loads(request.content)
        seen.append(body)
        if not body.get("stream_options", {}).get("include_usage"):
            usage = {}
        else:
            usage = {"prompt_tokens": 8, "completion_tokens": 2}
        if body.get("tools"):
            delta = {"role": "assistant", "tool_calls": [{
                "index": 0, "id": "c", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"})}}]}
        else:
            delta = {"role": "assistant",
                     "content": '```json\n{"status": "ok"}\n```'}
        chunks = [{"choices": [{"delta": delta}]}]
        if usage:
            chunks.append({"choices": [], "usage": usage})
        return httpx.Response(200, content=_sse(chunks))

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    report = asyncio.run(verify_endpoint(InferenceConfig(streaming=True), ["m"]))
    assert seen and all(b.get("stream") for b in seen)
    assert all(b["stream_options"]["include_usage"] for b in seen)
    assert report.ok


def test_streaming_probe_reports_http_failures(monkeypatch):
    """A streaming endpoint that rejects the request must surface as an
    error, not as a silently empty message."""
    import httpx

    from harness.agents.health import verify_endpoint
    from harness.config import InferenceConfig

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.path.endswith("/metrics"):
            return httpx.Response(404)
        return httpx.Response(400, text="streaming not supported")

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    report = asyncio.run(verify_endpoint(InferenceConfig(streaming=True), ["m"]))
    assert not report.ok
    assert any("400" in e for e in report.errors)


# -- resource pool sizes must be usable, not merely present -----------------


def test_non_positive_pool_and_parallelism_sizes_are_rejected():
    """`heavy_test: 0` builds an asyncio.Semaphore(0): every task that asks
    for the pool blocks forever. Refuse it at load time."""
    import pytest as _pytest
    from pydantic import ValidationError

    from harness.config import (
        InferenceConcurrencyConfig,
        ParallelismConfig,
        ResourcePoolsConfig,
    )

    for field in ("llm", "heavy_build", "heavy_test", "git_integration"):
        for bad in (0, -1):
            with _pytest.raises(ValidationError):
                ResourcePoolsConfig(**{field: bad})
    for field in ("max_parallel_tasks", "max_parallel_agent_runs"):
        for bad in (0, -1):
            with _pytest.raises(ValidationError):
                ParallelismConfig(**{field: bad})
    for bad in (0, -1):
        with _pytest.raises(ValidationError):
            InferenceConcurrencyConfig(max_requests=bad)

    # ...and valid sizes still load.
    pools = ResourcePoolsConfig(llm=4, heavy_build=1, heavy_test=2, git_integration=1)
    assert pools.heavy_test == 2
    # starvation_rounds legitimately disables the anti-starvation bump at 0.
    assert ParallelismConfig(starvation_rounds=0).starvation_rounds == 0


# -- the context budget must cover the whole request ------------------------


def test_context_budget_accounts_for_the_system_prompt(tmp_path):
    """The system prompt travels in the same request; a budget that ignores
    it puts every large prompt over the real limit."""
    from harness.context.builder import CHARS_PER_TOKEN, ContextBuilder
    from harness.context.project_context import ProjectContext
    from harness.orchestrator.state_machine import Role

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    system_prompt = "S" * 1500
    # A realistic project context: stable sections are never truncated (that
    # would both lose rules and fracture the shared prefix), so the budget is
    # only meetable when the dynamic sections can absorb the overflow.
    project = ProjectContext(
        name="p", goal="ship the thing", repository_path=str(tmp_path),
        base_branch="main", architecture_rules=["keep modules small"],
    )

    for budget_tokens in (1000, 5000, 50000):
        builder = ContextBuilder(prompts, input_budget_tokens=budget_tokens)
        prompt = builder.build_prompt(
            Role.DEVELOPER, project, extra="E" * 200000,
            system_prompt=system_prompt,
        )
        limit = budget_tokens * CHARS_PER_TOKEN
        total = len(prompt) + len(system_prompt)
        assert total <= limit, (
            f"budget {budget_tokens} tokens ({limit} chars): the request the "
            f"runner sends is {total} chars — the system prompt was not counted")


def test_streaming_probe_sees_reasoning_without_retaining_it(monkeypatch):
    """A thinking model that answers only in reasoning deltas must count as a
    working completion on the streaming path, as it does on the JSON path —
    and the harness must still never hold the reasoning text."""
    import httpx

    from harness.agents.health import verify_endpoint
    from harness.agents.openai_compat import stream_chat_completion
    from harness.config import InferenceConfig

    secret = "SECRET-CHAIN-OF-THOUGHT"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.path.endswith("/metrics"):
            return httpx.Response(404)
        body = json.loads(request.content)
        usage = {"prompt_tokens": 8, "completion_tokens": 2}
        if body.get("tools"):
            delta = {"role": "assistant", "tool_calls": [{
                "index": 0, "id": "c", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"})}}]}
        else:
            delta = {"role": "assistant", "reasoning_content": secret,
                     "content": '```json\n{"status": "ok"}\n```'}
        return _sse_response(httpx, [
            {"choices": [{"delta": delta}]},
            {"choices": [], "usage": usage},
        ])

    _patch_async_client(monkeypatch, httpx.MockTransport(handler))

    report = asyncio.run(verify_endpoint(InferenceConfig(streaming=True), ["m"]))
    assert report.ok
    assert report.models["m"].reasoning_content_seen, (
        "streaming probe reported no reasoning although the server emitted it")

    # The accumulator the agent loop runs on keeps the flag, never the text.
    async def accumulate():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://x/v1"
        ) as client:
            return await stream_chat_completion(client, {"model": "m"})

    message, status, _ = asyncio.run(accumulate())
    assert status == 200
    assert message["_reasoning_seen"] is True
    assert secret not in json.dumps(message), (
        "hidden reasoning text leaked into the accumulated message")


# -- final-review findings ---------------------------------------------------


def test_llm_task_keys_cannot_escape_into_paths():
    """task_key flows into worktree dirs, branch names and artifact paths;
    security must not depend on the prompt, so hostile keys are rejected at
    plan ingestion — for split tasks exactly as for planned ones."""
    import pytest as _pytest
    from pydantic import ValidationError

    from harness.artifacts.schemas import Diagnosis, PlannedTask

    for bad in ("../../../../tmp/x", "a/b", "a b", ".hidden", "", "-lead",
                "x" * 65, "a\x00b"):
        with _pytest.raises(ValidationError):
            PlannedTask(task_key=bad, title="t")
    with _pytest.raises(ValidationError):
        PlannedTask(task_key="T001", title="t", dependencies=["../up"])
    # split tasks are PlannedTask too, so the Diagnostician's replacements
    # go through the same gate
    with _pytest.raises(ValidationError):
        Diagnosis(recommendation="SPLIT",
                  split_tasks=[{"task_key": "../x", "title": "t"}])

    assert PlannedTask(task_key="T001", title="t").task_key == "T001"
    assert PlannedTask(task_key="T001-fix_2", title="t",
                       dependencies=["T000"]).dependencies == ["T000"]


def test_health_probes_leave_room_for_thinking_tokens():
    """With the reasoning parser active the model spends tokens on
    reasoning_content before the tool call / JSON; a tight probe cap
    truncates mid-thought and fails a healthy endpoint at startup."""
    import httpx

    from harness.agents.health import PROBE_MAX_TOKENS, verify_endpoint
    from harness.config import InferenceConfig

    assert PROBE_MAX_TOKENS >= 1024

    probe_budgets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.path.endswith("/metrics"):
            return httpx.Response(404)
        body = json.loads(request.content)
        probe_budgets.append(body["max_tokens"])
        usage = {"prompt_tokens": 8, "completion_tokens": 2}
        if body.get("tools"):
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant",
                "tool_calls": [{"id": "c", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"})}}]}}],
                "usage": usage})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant",
                                     "content": '```json\n{"status": "ok"}\n```'}}],
            "usage": usage})

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    import unittest.mock as um
    with um.patch.object(
        httpx, "AsyncClient",
        lambda *a, **k: real_client(*a, **{**k, "transport": transport}),
    ):
        report = asyncio.run(verify_endpoint(InferenceConfig(), ["m"]))

    assert report.ok
    # capability probes carry the roomy budget; only the prefix-cache
    # probes (which need no completion) stay tiny
    capability = [b for b in probe_budgets if b != 8]
    assert capability and all(b == PROBE_MAX_TOKENS for b in capability)


async def test_malformed_200_body_is_retried_in_session(tmp_path, monkeypatch):
    """A proxy or overloaded server returning 200 with an unparseable body
    is as transient as a 5xx: it must hit the in-session retry, not abort
    the session and discard the tool-loop history."""
    import httpx

    import harness.agents.openai_compat as oc
    from harness.agents.openai_compat import LocalOpenAICompatibleAgentRunner
    from harness.agents.profile import ResolvedAgentRunSpec
    from harness.config import InferenceConfig
    from harness.orchestrator.resources import PrefixAffinityGate

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, text="<html>gateway buffering</html>")
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": '```json\n{"task": "T1", "summary": "s"}\n```'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5}})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(oc, "shared_client",
                        lambda base_url, t, k=None: httpx.AsyncClient(
                            base_url=base_url, transport=transport))

    inference = InferenceConfig(transient_retries=2, transient_retry_base_delay=0.01)
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(2))
    spec = ResolvedAgentRunSpec(
        provider="openai-compatible", model="m", role="analyst", profile_id="x",
        profile_version="1", profile_hash="h", max_turns=4,
        cwd=str(tmp_path), repo_root=str(tmp_path), base_url="http://fake/v1",
        system_prompt="system", prompt="task",
    )
    result = await runner.run(spec)

    assert result.status == "COMPLETED", (
        f"malformed 200 body aborted the session: {result.error}")
    assert len(calls) == 2, "the malformed response was not retried in-session"


def test_timeout_report_salvages_output_when_a_descendant_holds_the_pipe(monkeypatch):
    """A re-setsid'd descendant escapes the killed group and keeps the pipe
    open; the drain then times out too. The report must still carry what
    was captured before the deadline — not empty output."""
    import harness.process as hp

    monkeypatch.setattr(hp, "TERM_GRACE_SECONDS", 1)
    result = hp.run_command(
        "echo diagnostic-line; setsid sleep 8 & sleep 30", cwd="/tmp", timeout=1
    )
    assert result.timed_out
    assert "diagnostic-line" in result.output, (
        "the timeout report lost the output captured before the deadline")
    assert "outside the killed group" in result.output


# -- round-11 findings -------------------------------------------------------


async def test_malformed_stream_is_retried_in_session(tmp_path, monkeypatch):
    """The streaming twin of the malformed-200 fix: an HTML body or a stream
    cut off before [DONE] must hit the in-session retry, not complete the
    turn with an empty/partial assistant message."""
    import httpx

    import harness.agents.openai_compat as oc
    from harness.agents.openai_compat import LocalOpenAICompatibleAgentRunner
    from harness.agents.profile import ResolvedAgentRunSpec
    from harness.config import InferenceConfig
    from harness.orchestrator.resources import PrefixAffinityGate

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:                      # 200 + HTML: no SSE at all
            return httpx.Response(200, text="<html>gateway buffering</html>")
        if len(calls) == 2:                      # truncated: chunks, no [DONE]
            return httpx.Response(200, content=_sse_body_without_done())
        return _sse_response(httpx, [
            {"choices": [{"delta": {
                "role": "assistant",
                "content": '```json\n{"task": "T1", "summary": "s"}\n```'}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
        ])

    def _sse_body_without_done():
        chunk = {"choices": [{"delta": {"role": "assistant", "content": "partial"}}]}
        return f"data: {json.dumps(chunk)}\n\n".encode()

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(oc, "shared_client",
                        lambda base_url, t, k=None: httpx.AsyncClient(
                            base_url=base_url, transport=transport))

    inference = InferenceConfig(streaming=True, transient_retries=3,
                                transient_retry_base_delay=0.01)
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(2))
    spec = ResolvedAgentRunSpec(
        provider="openai-compatible", model="m", role="analyst", profile_id="x",
        profile_version="1", profile_hash="h", max_turns=4,
        cwd=str(tmp_path), repo_root=str(tmp_path), base_url="http://fake/v1",
        system_prompt="system", prompt="task",
    )
    result = await runner.run(spec)

    assert result.status == "COMPLETED"
    assert len(calls) == 3, (
        f"malformed/truncated streams were not retried (calls={len(calls)})")
    assert "T1" in (result.output_text or ""), (
        "the session completed on a damaged stream's partial content")


def test_pin_ref_failure_is_loud(tmp_path):
    """If the conflict ref cannot be written, the caller must find out
    BEFORE deleting the commit's only other ref — check=False would let
    disposal proceed and leave tasks.task_commit gc-prunable."""
    from harness.git.repository import GitError, GitRepository

    git = GitRepository(tmp_path / "repo")
    git.init()
    (git.path / "f.txt").write_text("x\n")
    git.add_all()
    git.commit("c")
    commit = git.head_commit()

    with pytest.raises(GitError):
        git.pin_ref("refs/harness/conflicts/..invalid", commit)
    with pytest.raises(GitError):
        git.pin_ref("refs/harness/conflicts/ok", "a" * 40)  # nonexistent object


def test_stable_sections_exceeding_the_budget_fail_loudly(tmp_path):
    """When the system prompt + stable sections alone exceed the budget,
    nothing downstream ever truncates them — every request would be
    rejected for context length. Fail once at build time instead."""
    from harness.context.builder import ContextBudgetError, ContextBuilder
    from harness.context.project_context import ProjectContext
    from harness.orchestrator.state_machine import Role

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    builder = ContextBuilder(prompts, input_budget_tokens=1000)   # 4000 chars
    oversized = ProjectContext(
        name="p", goal="g" * 20000, repository_path=str(tmp_path),
        base_branch="main",
    )
    with pytest.raises(ContextBudgetError, match="context_profile"):
        builder.build_prompt(Role.DEVELOPER, oversized, system_prompt="S" * 1000)

    # a fitting project still builds — the gate only fires on real overflow
    fitting = ProjectContext(
        name="p", goal="ship it", repository_path=str(tmp_path),
        base_branch="main",
    )
    assert builder.build_prompt(Role.DEVELOPER, fitting, system_prompt="sys")


# -- round-12 findings -------------------------------------------------------


async def test_stream_with_one_damaged_event_is_retried(tmp_path, monkeypatch):
    """A malformed data event followed by valid chunks and [DONE] must still
    invalidate the stream: the lost fragment may be content or a tool-call
    argument piece, and skipping it completes the turn with a silently
    partial response."""
    import httpx

    import harness.agents.openai_compat as oc
    from harness.agents.openai_compat import LocalOpenAICompatibleAgentRunner
    from harness.agents.profile import ResolvedAgentRunSpec
    from harness.config import InferenceConfig
    from harness.orchestrator.resources import PrefixAffinityGate

    calls: list[int] = []
    good = {"choices": [{"delta": {
        "role": "assistant",
        "content": '```json\n{"task": "T1", "summary": "s"}\n```'}}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 5}}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            body = (
                "data: {\"choices\": [{\"delta\": {\"content\": \"lost frag"  # cut mid-JSON
                + "\n\n"
                + f"data: {json.dumps(good)}\n\n"
                + "data: [DONE]\n\n"
            ).encode()
            return httpx.Response(200, content=body)
        return _sse_response(httpx, [good, usage])

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(oc, "shared_client",
                        lambda base_url, t, k=None: httpx.AsyncClient(
                            base_url=base_url, transport=transport))

    inference = InferenceConfig(streaming=True, transient_retries=2,
                                transient_retry_base_delay=0.01)
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(2))
    spec = ResolvedAgentRunSpec(
        provider="openai-compatible", model="m", role="analyst", profile_id="x",
        profile_version="1", profile_hash="h", max_turns=4,
        cwd=str(tmp_path), repo_root=str(tmp_path), base_url="http://fake/v1",
        system_prompt="system", prompt="task",
    )
    result = await runner.run(spec)

    assert result.status == "COMPLETED"
    assert len(calls) == 2, (
        "a stream with a damaged event was accepted instead of retried")


def test_dynamic_sections_are_squeezed_below_floor_before_rejecting(tmp_path):
    """The overflow error is for STABLE content only: while trimmable text
    remains — even below the preferred readable floor — the prompt must be
    squeezed to fit, not rejected."""
    from harness.context.builder import CHARS_PER_TOKEN, ContextBuilder
    from harness.context.project_context import ProjectContext
    from harness.orchestrator.state_machine import Role

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    budget_tokens = 1000                                    # 4000 chars
    builder = ContextBuilder(prompts, input_budget_tokens=budget_tokens)
    # stable content close to (but under) the budget + a large dynamic extra:
    # the preferred floor alone would overflow, full squeeze fits.
    project = ProjectContext(
        name="p", goal="g" * 2500, repository_path=str(tmp_path),
        base_branch="main",
    )
    prompt = builder.build_prompt(
        Role.DEVELOPER, project, extra="E" * 50000, system_prompt="S" * 200)
    assert len(prompt) + 200 <= budget_tokens * CHARS_PER_TOKEN, (
        "squeezable prompt was not fitted into the budget")


async def test_context_budget_error_fails_the_project_durably(config, monkeypatch):
    """ContextBudgetError at dispatch must record PROJECT_FAILED — not
    escape as a traceback that leaves the project stuck in PLANNING and
    recurs identically on the next invocation."""
    from harness.context.builder import ContextBudgetError
    from harness.orchestrator.project import ProjectOrchestrator
    from harness.orchestrator.state_machine import ProjectState

    orchestrator = ProjectOrchestrator(config)

    async def exploding_invoke(*args, **kwargs):
        raise ContextBudgetError("stable prompt content exceeds the input budget")

    monkeypatch.setattr(orchestrator.invoker, "invoke", exploding_invoke)
    state = await orchestrator.run()

    assert state == ProjectState.FAILED
    row = orchestrator.projects.get(1)
    assert row["status"] == ProjectState.FAILED.value
    failed = orchestrator.db.query_all(
        "SELECT * FROM events WHERE event_type = 'PROJECT_FAILED'")
    assert failed and "context budget" in (failed[0]["payload"] or "")
