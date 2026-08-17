"""End-to-end orchestrator tests with a scripted fake agent runner.

Verifies the deterministic wiring: Planner -> Analyst -> Developer ->
Tester -> Verification -> Reviewer -> Commit, plus the REPAIR cycle and
the Diagnostician escalation — no LLM involved.
"""

import json
from pathlib import Path

import pytest

import harness.orchestrator.agent_invoker as agent_invoker_module
from harness.agents.base import AgentRequest, AgentResult
from harness.config import HarnessConfig, ProjectConfig, VerificationConfig
from harness.git.repository import GitRepository
from harness.orchestrator.project import ProjectOrchestrator
from harness.orchestrator.state_machine import ProjectState, Role, TaskState

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeRunner:
    """Returns scripted JSON per role; the Developer really edits the repo."""

    def __init__(self, repo_path: Path, review_verdicts: list[str]):
        self.repo_path = repo_path
        self.review_verdicts = review_verdicts
        self.calls: list[Role] = []

    async def run(self, request: AgentRequest) -> AgentResult:
        self.calls.append(request.role)
        payload = self._payload(request.role)
        return AgentResult(
            status="COMPLETED",
            output_text=f"```json\n{json.dumps(payload)}\n```",
            session_id=f"session-{len(self.calls)}",
            cost_usd=0.01,
        )

    def _payload(self, role: Role) -> dict:
        if role == Role.PLANNER:
            return {
                "summary": "one task",
                "tasks": [
                    {
                        "task_key": "T001",
                        "title": "write feature file",
                        "goal": "create feature.txt",
                        "acceptance_criteria": ["feature.txt exists"],
                        "dependencies": [],
                    }
                ],
            }
        if role == Role.ANALYST:
            return {"task": "T001", "summary": "create the file", "files": ["feature.txt"]}
        if role == Role.DEVELOPER:
            (self.repo_path / "feature.txt").write_text("feature\n")
            return {"task": "T001", "summary": "wrote file", "changed_files": ["feature.txt"]}
        if role == Role.TESTER:
            return {"task": "T001", "summary": "coverage ok"}
        if role == Role.REVIEWER:
            verdict = self.review_verdicts.pop(0) if self.review_verdicts else "PASS"
            return {
                "verdict": verdict,
                "summary": "review",
                "blocking_issues": []
                if verdict == "PASS"
                else [{"file": "feature.txt", "issue": "x", "required_fix": "y"}],
            }
        if role == Role.DIAGNOSTICIAN:
            return {"root_causes": ["scope"], "recommendation": "BLOCKED"}
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
        project=ProjectConfig(name="test-project", goal="make feature"),
        workspace_dir=str(workspace),
        verification=VerificationConfig(language="none", commands=["true"]),
    )
    cfg.config_path = REPO_ROOT / "config.yaml"  # prompts_path -> real prompts/
    return cfg


def install_fake(monkeypatch, fake: FakeRunner) -> None:
    monkeypatch.setattr(agent_invoker_module, "create_runner", lambda provider: fake)


async def test_happy_path_completes_project(config, monkeypatch):
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.COMPLETED
    assert fake.calls == [Role.PLANNER, Role.ANALYST, Role.DEVELOPER, Role.TESTER, Role.REVIEWER]
    task = orchestrator.tasks.get_by_key(1, "T001")
    assert task["status"] == "COMPLETED"
    assert task["current_commit"]
    assert "agent(T001): write feature file" in orchestrator.git.log_oneline()
    assert not orchestrator.git.is_dirty()
    brief = orchestrator.artifacts.load_json(
        orchestrator.artifacts.task_artifact_path("T001", "task-brief.json")
    )
    assert brief["task"] == "T001"
    project = orchestrator.projects.get(1)
    assert project["spent_usd"] > 0


async def test_repair_cycle_then_pass(config, monkeypatch):
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["REPAIR", "PASS"])
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.COMPLETED
    # analyst once, developer/tester/reviewer twice (fresh sessions per attempt)
    assert fake.calls.count(Role.ANALYST) == 1
    assert fake.calls.count(Role.DEVELOPER) == 2
    assert fake.calls.count(Role.REVIEWER) == 2
    task = orchestrator.tasks.get_by_key(1, "T001")
    assert task["attempt_count"] == 2


async def test_repeated_repair_escalates_to_diagnostician(config, monkeypatch):
    config.limits.max_attempts = 2
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(
        config.repository_path, review_verdicts=["REPAIR", "REPAIR", "REPAIR", "REPAIR"]
    )
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert Role.DIAGNOSTICIAN in fake.calls
    task = orchestrator.tasks.get_by_key(1, "T001")
    assert task["status"] == "BLOCKED"          # diagnostician said BLOCKED
    assert state == ProjectState.BLOCKED        # project can't finish with blocked work
    assert not orchestrator.git.is_dirty()      # failed work was discarded
    # diagnostician runs are attached to an attempt so they count toward
    # max_agent_runs_per_task (Codex P2)
    diag_run = orchestrator.db.query_one(
        "SELECT attempt_id FROM agent_runs WHERE role = 'diagnostician'"
    )
    assert diag_run is not None and diag_run["attempt_id"] is not None


async def test_analyst_failure_counts_toward_retry_limit(config, monkeypatch):
    """A failing Analyst must burn attempts, not loop forever (Codex P1)."""
    config.limits.max_attempts = 2
    monkeypatch.setattr(agent_invoker_module, "TECHNICAL_RETRY_DELAY", 0.0)
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    original_run = fake.run

    async def run_failing_analyst(request):
        if request.role in (Role.ANALYST, Role.DIAGNOSTICIAN):
            fake.calls.append(request.role)
            return AgentResult(status="FAILED", error="provider down")
        return await original_run(request)

    fake.run = run_failing_analyst
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    task = orchestrator.tasks.get_by_key(1, "T001")
    assert task["status"] == "BLOCKED"          # diagnosis also failed -> BLOCKED
    assert task["attempt_count"] >= config.limits.max_attempts  # failures were recorded
    assert state == ProjectState.BLOCKED


async def test_deadlocked_plan_blocks_project(config, monkeypatch):
    """A plan whose dependencies can never be satisfied must not complete (Codex P1)."""
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=[])
    original_payload = fake._payload

    def payload_with_bad_dependency(role):
        if role == Role.PLANNER:
            return {
                "summary": "broken plan",
                "tasks": [
                    {
                        "task_key": "T001",
                        "title": "unreachable task",
                        "goal": "depends on a task that does not exist",
                        "dependencies": ["T999"],
                    }
                ],
            }
        return original_payload(role)

    fake._payload = payload_with_bad_dependency
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.BLOCKED        # never COMPLETED with pending work
    task = orchestrator.tasks.get_by_key(1, "T001")
    assert task["status"] == "PENDING"


async def test_agent_run_budget_cap_enforced(config, monkeypatch):
    """A single run exceeding agent_run_usd fails the attempt (Codex P2)."""
    config.budget.agent_run_usd = 0.005  # below FakeRunner's 0.01 per run
    config.limits.max_attempts = 1
    monkeypatch.setattr(agent_invoker_module, "TECHNICAL_RETRY_DELAY", 0.0)
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    # planner run itself blows the cap -> project fails fast, never COMPLETED
    assert state != ProjectState.COMPLETED


async def test_diagnostician_sees_diff_after_agent_failure(config, monkeypatch):
    """The Diagnostician must receive the failed attempt's diff even when the
    failure was an aborted agent run (Codex P1)."""
    config.limits.max_attempts = 1
    monkeypatch.setattr(agent_invoker_module, "TECHNICAL_RETRY_DELAY", 0.0)
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=[])
    prompts: dict[Role, list[str]] = {}
    original_run = fake.run

    async def recording_run(request):
        prompts.setdefault(request.role, []).append(request.prompt)
        if request.role == Role.TESTER:  # dies AFTER the Developer edited files
            fake.calls.append(request.role)
            return AgentResult(status="FAILED", error="tester crashed")
        return await original_run(request)

    fake.run = recording_run
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.BLOCKED  # fake diagnostician recommends BLOCKED
    diag_prompts = prompts.get(Role.DIAGNOSTICIAN)
    assert diag_prompts, "diagnostician was never invoked"
    assert "feature.txt" in diag_prompts[0]      # the diff made it into the context
    assert not orchestrator.git.is_dirty()       # and the tree was still reset


async def test_budget_pause_leaves_clean_tree(config, monkeypatch):
    """PAUSED must leave the repo at the last good commit so the next startup
    can recover; the dirty diff is archived first (Codex P1)."""
    config.budget.project_usd = 0.025  # exhausted right after the Developer run
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.PAUSED
    assert not orchestrator.git.is_dirty()  # developer's half-done work was reset...
    archived = list((orchestrator.artifacts.root / "diagnostics").glob("T001-budget-paused.diff"))
    assert len(archived) == 1               # ...but archived, not lost
    assert "feature" in archived[0].read_text()
    # a rerun with a raised budget starts cleanly instead of being refused
    config.budget.project_usd = 100.0
    orchestrator2 = ProjectOrchestrator(config)
    fake2 = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    install_fake(monkeypatch, fake2)
    assert await orchestrator2.run() == ProjectState.COMPLETED


async def test_replan_updates_and_drops_existing_tasks(config, monkeypatch):
    """A revised plan must replace unfinished task definitions, not be discarded (Codex P1)."""
    orchestrator = ProjectOrchestrator(config)
    project = orchestrator.ensure_project()
    pid = project["id"]
    orchestrator.tasks.create(pid, "T001", "old title", "old goal")
    orchestrator.tasks.create(pid, "T002", "obsolete task")
    t3 = orchestrator.tasks.create(pid, "T003", "already done")
    orchestrator.tasks.set_status(t3, TaskState.COMPLETED, force=True)

    fake = FakeRunner(config.repository_path, review_verdicts=[])

    def replan_payload(role):
        assert role == Role.PLANNER
        return {
            "summary": "revised",
            "tasks": [
                {"task_key": "T000", "title": "new prerequisite"},
                {"task_key": "T001", "title": "new title", "goal": "new goal",
                 "dependencies": ["T000"]},
                {"task_key": "T003", "title": "must not touch completed"},
            ],
        }

    fake._payload = replan_payload
    install_fake(monkeypatch, fake)

    await orchestrator._plan(pid, orchestrator.project_context(), replan=True)

    t1 = orchestrator.tasks.get_by_key(pid, "T001")
    assert t1["title"] == "new title"
    assert json.loads(t1["dependencies"]) == ["T000"]          # revised definition applied
    assert orchestrator.tasks.get_by_key(pid, "T000") is not None  # new task inserted
    assert orchestrator.tasks.get_by_key(pid, "T002")["status"] == "SKIPPED"  # dropped
    assert orchestrator.tasks.get_by_key(pid, "T003")["title"] == "already done"  # untouched


async def test_failed_run_cost_still_hits_run_cap(config, monkeypatch):
    """Failed runs spend money too; the per-run cap must stop retries (Codex P2)."""
    config.budget.agent_run_usd = 1.0
    monkeypatch.setattr(agent_invoker_module, "TECHNICAL_RETRY_DELAY", 0.0)
    orchestrator = ProjectOrchestrator(config)
    calls = []

    class ExpensiveFailingRunner:
        async def run(self, request):
            calls.append(request.role)
            return AgentResult(status="FAILED", error="boom", cost_usd=5.0)

    monkeypatch.setattr(agent_invoker_module, "create_runner",
                        lambda provider: ExpensiveFailingRunner())

    state = await orchestrator.run()

    assert state == ProjectState.FAILED   # planner attempt aborted
    assert len(calls) == 1                # cap raised immediately — no blind retries
    project = orchestrator.projects.get(1)
    assert project["spent_usd"] == pytest.approx(5.0)  # cost still recorded


async def test_verification_failure_triggers_fresh_developer(config, monkeypatch):
    config.verification.commands = ["test -f marker.txt"]  # fails until 2nd attempt
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    original_payload = fake._payload

    def payload_with_marker(role):
        if role == Role.DEVELOPER and fake.calls.count(Role.DEVELOPER) >= 2:
            (config.repository_path / "marker.txt").write_text("ok\n")
        return original_payload(role)

    fake._payload = payload_with_marker
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.COMPLETED
    assert fake.calls.count(Role.DEVELOPER) == 2
    # reviewer only ran once — verification failure short-circuits to Developer
    assert fake.calls.count(Role.REVIEWER) == 1
