"""Parallel scheduling end-to-end with a scripted fake runner: worktree
isolation, dependency gating, parallel limits, serialized integration,
conflict -> fresh repair, and parallel crash recovery (spec item 71)."""

import asyncio
import json
import re
from pathlib import Path

import pytest

import harness.orchestrator.agent_invoker as agent_invoker_module
from harness.agents.base import AgentResult, BaseAgentRunner
from harness.agents.profile import ResolvedAgentRunSpec
from harness.config import HarnessConfig, ProjectConfig, VerificationConfig
from harness.git.repository import GitRepository
from harness.orchestrator.project import ProjectOrchestrator
from harness.orchestrator.state_machine import ProjectState, Role

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_KEY_RE = re.compile(r"Task: (T\d+)")


class ParallelFakeRunner(BaseAgentRunner):
    """Task-aware scripted runner. Tracks per-task worktrees and overlap."""

    provider_name = "fake"

    def __init__(self, task_keys: list[str], dependencies: dict[str, list[str]] | None = None,
                 conflict_file: str | None = None):
        self.task_keys = task_keys
        self.dependencies = dependencies or {}
        self.conflict_file = conflict_file
        self.calls: list[tuple[str, Role, str]] = []   # (task_key, role, cwd)
        self.active = 0
        self.peak_active = 0
        self.developer_rounds: dict[str, int] = {}

    def _task_key(self, spec: ResolvedAgentRunSpec) -> str:
        match = TASK_KEY_RE.search(spec.prompt)
        return match.group(1) if match else "PROJECT"

    async def run(self, spec: ResolvedAgentRunSpec) -> AgentResult:
        role = Role(spec.role)
        task_key = self._task_key(spec)
        cwd = Path(spec.cwd)
        self.calls.append((task_key, role, str(cwd)))
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        await asyncio.sleep(0.02)  # force real overlap between task coroutines
        self.active -= 1

        payload = self._payload(role, task_key, cwd)
        return AgentResult(
            status="COMPLETED",
            output_text=f"```json\n{json.dumps(payload)}\n```",
            cost_usd=0.01,
        )

    def _payload(self, role: Role, task_key: str, cwd: Path) -> dict:
        if role == Role.PLANNER:
            return {
                "summary": "plan",
                "tasks": [
                    {"task_key": key, "title": f"task {key}", "goal": f"do {key}",
                     "acceptance_criteria": [], "dependencies": self.dependencies.get(key, [])}
                    for key in self.task_keys
                ],
            }
        if role == Role.ANALYST:
            return {"task": task_key, "summary": "brief", "files": []}
        if role == Role.DEVELOPER:
            self.developer_rounds[task_key] = self.developer_rounds.get(task_key, 0) + 1
            (cwd / f"{task_key}.txt").write_text(f"work by {task_key}\n")
            if self.conflict_file:
                (cwd / self.conflict_file).write_text(f"content from {task_key}\n")
            return {"task": task_key, "summary": "impl", "changed_files": [f"{task_key}.txt"]}
        if role == Role.TESTER:
            return {"task": task_key, "summary": "tests ok"}
        if role == Role.REVIEWER:
            return {"verdict": "PASS", "summary": "ok", "blocking_issues": []}
        if role == Role.DIAGNOSTICIAN:
            return {"root_causes": [], "recommendation": "BLOCKED"}
        raise AssertionError(f"unexpected role {role}")


@pytest.fixture
def config(tmp_path) -> HarnessConfig:
    workspace = tmp_path / "workspace"
    repo = GitRepository(workspace / "repository")
    repo.init()
    (repo.path / "README.md").write_text("seed\n")
    repo.add_all()
    repo.commit("initial")

    cfg = HarnessConfig(
        project=ProjectConfig(name="par-project", goal="parallel work"),
        workspace_dir=str(workspace),
        verification=VerificationConfig(language="none", commands=["true"]),
    )
    cfg.config_path = REPO_ROOT / "config.yaml"
    return cfg


def install(monkeypatch, runner) -> None:
    monkeypatch.setattr(agent_invoker_module, "create_runner",
                        lambda provider, inference=None, llm_gate=None: runner)
    monkeypatch.setattr(agent_invoker_module, "TECHNICAL_RETRY_DELAY", 0.0)


# ---------------------------------------------------------------------------


async def test_parallel_tasks_isolated_worktrees_and_limits(config, monkeypatch):
    keys = [f"T{i:03d}" for i in range(1, 7)]
    config.parallelism.max_parallel_tasks = 3
    runner = ParallelFakeRunner(keys)
    install(monkeypatch, runner)
    orchestrator = ProjectOrchestrator(config)

    state = await orchestrator.run()
    assert state == ProjectState.COMPLETED

    # every task completed with its own integration commit
    for task in orchestrator.tasks.list_for_project(1):
        assert task["status"] == "COMPLETED"
        assert task["task_commit"] and task["integration_commit"]
        assert (orchestrator.git.path / f"{task['task_key']}.txt").exists()

    # different tasks used different worktrees; roles of one task shared one
    cwds_by_task: dict[str, set[str]] = {}
    for task_key, role, cwd in runner.calls:
        if task_key == "PROJECT":
            continue
        cwds_by_task.setdefault(task_key, set()).add(cwd)
    for task_key, cwds in cwds_by_task.items():
        assert len(cwds) == 1, f"{task_key} roles saw multiple worktrees: {cwds}"
    all_cwds = [next(iter(c)) for c in cwds_by_task.values()]
    assert len(set(all_cwds)) == len(all_cwds), "tasks shared a worktree"
    for cwd in all_cwds:
        assert cwd != str(config.repository_path)

    # genuine overlap happened, and never above the configured limit
    assert runner.peak_active > 1
    assert runner.peak_active <= config.parallelism.max_parallel_tasks

    # worktrees are cleaned up after completion
    assert orchestrator.worktrees.registered_paths() == []

    # ledger and state survived parallel writes intact
    assert orchestrator.events.find_stream_gaps() == []
    from harness.orchestrator.invariants import Severity
    errors = [v for v in orchestrator.invariants.check_all(1)
              if v.severity == Severity.ERROR]
    assert errors == []

    # ContextManifests: distinct per run and resolvable on disk
    runs = orchestrator.db.query_all(
        "SELECT id, context_manifest_path FROM agent_runs WHERE status = 'COMPLETED'")
    paths = [r["context_manifest_path"] for r in runs]
    assert len(paths) == len(set(paths))
    for stored in paths:
        assert orchestrator.artifacts.resolve(stored).exists()


async def test_sixteen_ready_tasks_schedule_in_parallel(config, monkeypatch):
    """The target production shape: 16 independent READY tasks, 16 slots."""
    keys = [f"T{i:03d}" for i in range(1, 17)]
    config.parallelism.max_parallel_tasks = 16
    runner = ParallelFakeRunner(keys)
    install(monkeypatch, runner)
    orchestrator = ProjectOrchestrator(config)

    state = await orchestrator.run()
    assert state == ProjectState.COMPLETED
    tasks = orchestrator.tasks.list_for_project(1)
    assert len(tasks) == 16
    assert all(t["status"] == "COMPLETED" for t in tasks)
    assert runner.peak_active > 4            # genuinely parallel
    assert runner.peak_active <= 16
    assert orchestrator.events.find_stream_gaps() == []
    assert orchestrator.worktrees.registered_paths() == []


async def test_dependent_task_waits_for_prerequisite(config, monkeypatch):
    config.parallelism.max_parallel_tasks = 4
    runner = ParallelFakeRunner(["T001", "T002"], dependencies={"T002": ["T001"]})
    install(monkeypatch, runner)
    orchestrator = ProjectOrchestrator(config)

    state = await orchestrator.run()
    assert state == ProjectState.COMPLETED

    sequence = [(k, r) for k, r, _ in runner.calls if k != "PROJECT"]
    t001_last = max(i for i, (k, _) in enumerate(sequence) if k == "T001")
    t002_first = min(i for i, (k, _) in enumerate(sequence) if k == "T002")
    assert t001_last < t002_first, "T002 started before its dependency finished"


async def test_integration_conflict_routes_to_fresh_repair(config, monkeypatch):
    """Two independent tasks add the SAME file from the same base commit:
    the second integration hits an add/add conflict, the task moves through
    INTEGRATION_CONFLICT into a fresh repair attempt on the new base, and
    completes on the second round."""
    config.parallelism.max_parallel_tasks = 2
    runner = ParallelFakeRunner(["T001", "T002"], conflict_file="shared.txt")
    install(monkeypatch, runner)
    orchestrator = ProjectOrchestrator(config)

    state = await orchestrator.run()
    assert state == ProjectState.COMPLETED

    conflict_events = orchestrator.db.query_all(
        "SELECT * FROM events WHERE event_type = 'INTEGRATION_CONFLICT'")
    assert len(conflict_events) == 1

    # the conflicted task ran a second developer round (fresh repair attempt)
    assert sorted(runner.developer_rounds.values()) == [1, 2]

    for task in orchestrator.tasks.list_for_project(1):
        assert task["status"] == "COMPLETED"
        assert task["integration_commit"]
    # both integrations landed; the file exists with ONE of the contents
    content = (orchestrator.git.path / "shared.txt").read_text()
    assert content.startswith("content from T")
    # the failed integration op is settled, nothing PENDING
    assert orchestrator.operations.unfinished() == []
    assert orchestrator.worktrees.registered_paths() == []


async def test_parallel_crash_recovery_recovers_every_running_task(config, monkeypatch):
    """Crash with TWO tasks mid-flight in separate worktrees: startup
    recovery must settle BOTH (archive + remove worktree, interrupt
    attempt, requeue task) — not just one."""
    runner = ParallelFakeRunner(["T001", "T002"])
    install(monkeypatch, runner)
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]

    # simulate the crashed state by hand: two worktrees with dirty files,
    # RUNNING attempts, RUNNING agent runs
    tid1 = orchestrator.tasks.create(pid, "T001", "one")
    tid2 = orchestrator.tasks.create(pid, "T002", "two")
    attempts = []
    for tid, key in ((tid1, "T001"), (tid2, "T002")):
        env = orchestrator.task_runner._create_env(key, 1)
        (env.handle.path / "half-done.txt").write_text(f"wip {key}\n")
        aid = orchestrator.tasks.start_attempt(
            tid, env.git.head_commit(),
            worktree_path=str(env.handle.path), branch=env.handle.branch)
        orchestrator.tasks.set_status(tid, __import__(
            "harness.orchestrator.state_machine", fromlist=["TaskState"]
        ).TaskState.EXECUTING, force=True)
        orchestrator.runs.start_run(pid, "developer", aid, mutating=True)
        attempts.append(aid)

    acted = orchestrator.recovery.recover(orchestrator.projects.get(pid))
    assert acted

    for aid in attempts:
        assert orchestrator.tasks.get_attempt(aid)["status"] == "INTERRUPTED"
    for tid in (tid1, tid2):
        assert orchestrator.tasks.get(tid)["status"] == "READY"
    assert orchestrator.worktrees.registered_paths() == []
    assert orchestrator.runs.running_runs() == []
    # both dirty diffs were archived before removal
    diagnostics = list((config.artifacts_path / "diagnostics").glob("interrupted-*.diff"))
    assert len(diagnostics) == 2


async def test_integration_intent_reconciled_after_crash(config, monkeypatch):
    """Crash between the integration merge and its DB record: recovery finds
    the merge commit by Operation-Id trailer and completes the task — the
    merge is never executed twice."""
    runner = ParallelFakeRunner(["T001"])
    install(monkeypatch, runner)
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    tid = orchestrator.tasks.create(pid, "T001", "one")

    env = orchestrator.task_runner._create_env("T001", 1)
    aid = orchestrator.tasks.start_attempt(
        tid, env.git.head_commit(),
        worktree_path=str(env.handle.path), branch=env.handle.branch)
    (env.handle.path / "done.txt").write_text("finished\n")
    task_commit = None
    env.git.add_all()
    task_commit = env.git.commit("agent(T001): one")
    orchestrator.tasks.set_task_commit(tid, task_commit)
    orchestrator.tasks.finish_attempt(
        aid, __import__("harness.orchestrator.state_machine",
                        fromlist=["AttemptState"]).AttemptState.PASSED)
    orchestrator.tasks.set_status(tid, __import__(
        "harness.orchestrator.state_machine", fromlist=["TaskState"]
    ).TaskState.INTEGRATING, force=True)
    outcome = await orchestrator.integration.integrate(
        task_key="T001", task_commit=task_commit, title="one",
        project_id=pid, task_id=tid, attempt_id=aid)
    assert outcome.status.value == "MERGED"
    # crash: pending_operation_id never settled, no task update
    orchestrator.worktrees.remove(env.handle)

    head_before = orchestrator.git.head_commit()
    acted = orchestrator.recovery.recover(orchestrator.projects.get(pid))
    assert acted
    assert orchestrator.git.head_commit() == head_before   # no double merge
    task = orchestrator.tasks.get(tid)
    assert task["status"] == "COMPLETED"
    assert task["integration_commit"] == outcome.integration_commit
    op = orchestrator.operations.get(outcome.operation_id)
    assert op["status"] == "RECONCILED"
