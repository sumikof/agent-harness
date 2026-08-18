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
                        lambda provider, inference=None, llm_gate=None: runner)

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
