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
from harness.orchestrator.state_machine import ProjectState, Role

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
