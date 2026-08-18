"""End-to-end orchestrator tests with a scripted fake agent runner.

Verifies the deterministic wiring: Planner -> Analyst -> Developer ->
Tester -> Verification -> Reviewer -> Commit, plus the REPAIR cycle and
the Diagnostician escalation — no LLM involved.
"""

import json
from pathlib import Path

import pytest

import harness.orchestrator.agent_invoker as agent_invoker_module
from harness.agents.base import AgentResult, BaseAgentRunner
from harness.agents.profile import ResolvedAgentRunSpec
from harness.config import HarnessConfig, ProjectConfig, VerificationConfig
from harness.git.repository import GitRepository
from harness.orchestrator.project import ProjectOrchestrator
from harness.orchestrator.state_machine import ProjectState, Role, TaskState

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeRunner(BaseAgentRunner):
    """Returns scripted JSON per role; the Developer really edits the repo.

    Inherits capabilities()/resolve() from BaseAgentRunner, so run()
    receives the ResolvedAgentRunSpec the invoker persisted beforehand.
    """

    provider_name = "fake"

    def __init__(self, repo_path: Path, review_verdicts: list[str]):
        self.repo_path = repo_path
        self.review_verdicts = review_verdicts
        self.calls: list[Role] = []
        self.cwd: Path = repo_path  # updated per run: the task's worktree

    async def run(self, request: ResolvedAgentRunSpec) -> AgentResult:
        self.calls.append(request.role)
        self.cwd = Path(request.cwd)
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
            (self.cwd / "feature.txt").write_text("feature\n")
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
    monkeypatch.setattr(agent_invoker_module, "create_runner", lambda provider, inference=None: fake)


def test_ensure_project_identity_rules(config, tmp_path):
    """Identity may be corrected while nothing has run; once work exists a
    changed repository must fail loudly instead of reusing stale state
    (Codex P1 + P2)."""
    ProjectOrchestrator(config).ensure_project()

    other_repo = GitRepository(tmp_path / "other-repo")
    other_repo.init()
    (other_repo.path / "x.txt").write_text("x\n")
    other_repo.add_all()
    other_repo.commit("init")

    changed = config.model_copy(deep=True)
    changed.config_path = config.config_path
    changed.project.repository = str(other_repo.path)  # same name, new repo

    # untouched project (CREATED, no tasks): treated as fixing a typo
    orchestrator = ProjectOrchestrator(changed)
    row = orchestrator.ensure_project()
    assert row["repository"] == str(other_repo.path)

    # once work exists, identity is frozen
    orchestrator.tasks.create(row["id"], "T001", "some work")
    back_to_original = config.model_copy(deep=True)
    back_to_original.config_path = config.config_path
    with pytest.raises(RuntimeError, match="stale state"):
        ProjectOrchestrator(back_to_original).ensure_project()

    # a goal change alone is content, not identity — accepted and persisted
    changed_goal = changed.model_copy(deep=True)
    changed_goal.config_path = config.config_path
    changed_goal.project.goal = "refined goal"
    row = ProjectOrchestrator(changed_goal).ensure_project()
    assert row["goal"] == "refined goal"


def test_ensure_project_rejects_wrong_branch(config):
    """Harness checkpoints go to the checked-out branch, so it must match
    base_branch at startup (Codex P1)."""
    repo = GitRepository(config.repository_path)
    repo._run("checkout", "-b", "feature")
    with pytest.raises(RuntimeError, match="base_branch"):
        ProjectOrchestrator(config).ensure_project()


async def test_empty_plan_fails_project_instead_of_completing(config, monkeypatch):
    """A planner returning {\"tasks\": []} must not let the project complete
    with zero work (Codex P1)."""
    monkeypatch.setattr(agent_invoker_module, "TECHNICAL_RETRY_DELAY", 0.0)
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=[])
    original_payload = fake._payload

    def empty_plan(role):
        if role == Role.PLANNER:
            return {"summary": "nothing to do", "tasks": []}
        return original_payload(role)

    fake._payload = empty_plan
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.FAILED  # loud failure, not a silent COMPLETED


async def test_changed_goal_triggers_replan(config, monkeypatch):
    """Resuming with a new goal must revise the stored plan before executing
    it (Codex P1)."""
    orchestrator = ProjectOrchestrator(config)
    row = orchestrator.ensure_project()
    pid = row["id"]
    orchestrator.tasks.create(pid, "T001", "task for the old goal")
    orchestrator.projects.set_status(pid, ProjectState.PAUSED, force=True)

    changed = config.model_copy(deep=True)
    changed.config_path = config.config_path
    changed.project.goal = "brand new goal"
    orchestrator2 = ProjectOrchestrator(changed)
    fake = FakeRunner(changed.repository_path, review_verdicts=["PASS"])
    original_payload = fake._payload

    def revised_plan(role):
        if role == Role.PLANNER:
            return {
                "summary": "revised for the new goal",
                "tasks": [
                    {
                        "task_key": "T001",
                        "title": "revised for new goal",
                        "goal": "aligned with the new goal",
                        "acceptance_criteria": ["feature.txt exists"],
                        "dependencies": [],
                    }
                ],
            }
        return original_payload(role)

    fake._payload = revised_plan
    install_fake(monkeypatch, fake)

    state = await orchestrator2.run()

    assert state == ProjectState.COMPLETED
    assert Role.PLANNER in fake.calls  # replan actually happened
    task = orchestrator2.tasks.get_by_key(pid, "T001")
    assert task["title"] == "revised for new goal"


async def test_goal_change_on_finished_project_rejected(config, monkeypatch):
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(config.repository_path, review_verdicts=["PASS"])
    install_fake(monkeypatch, fake)
    assert await orchestrator.run() == ProjectState.COMPLETED

    changed = config.model_copy(deep=True)
    changed.config_path = config.config_path
    changed.project.goal = "a different goal"
    with pytest.raises(RuntimeError, match="COMPLETED"):
        ProjectOrchestrator(changed).ensure_project()


def test_ensure_project_creates_baseline_commit(tmp_path):
    """A commitless repo gets a baseline commit at project start, so every
    later reset/recovery has a HEAD; ignored user data is untouched (Codex P1)."""
    workspace = tmp_path / "workspace"
    repo = GitRepository(workspace / "repository")
    repo.init()
    (repo.path / "existing.txt").write_text("pre-existing\n")
    (repo.path / ".git" / "info" / "exclude").write_text("private/\n")
    (repo.path / "private").mkdir()
    (repo.path / "private" / "data.txt").write_text("user data\n")

    cfg = HarnessConfig(
        project=ProjectConfig(name="baseline-test", goal="g"),
        workspace_dir=str(workspace),
        verification=VerificationConfig(language="none", commands=["true"]),
    )
    cfg.config_path = REPO_ROOT / "config.yaml"
    orchestrator = ProjectOrchestrator(cfg)
    orchestrator.ensure_project()

    assert repo.head_commit() is not None       # baseline exists
    assert not repo.is_dirty()                  # existing.txt was committed
    assert (repo.path / "private" / "data.txt").exists()  # ignored data kept
    # and a discard after agent work now really resets
    (repo.path / "existing.txt").write_text("agent half-done\n")
    orchestrator.checkpoint.discard_working_tree()
    assert (repo.path / "existing.txt").read_text() == "pre-existing\n"


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
    # Artifacts now carry a provenance envelope around the domain payload.
    assert brief["artifact_type"] == "TaskBrief"
    assert brief["producer"]["role"] == "analyst"
    assert brief["payload"]["task"] == "T001"
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


async def test_empty_split_diagnosis_blocks_instead_of_dropping(config, monkeypatch):
    """SPLIT with no replacement tasks must not silently drop the work (Codex P1)."""
    config.limits.max_attempts = 1
    orchestrator = ProjectOrchestrator(config)
    fake = FakeRunner(
        config.repository_path, review_verdicts=["REPAIR", "REPAIR", "REPAIR"]
    )
    original_payload = fake._payload

    def split_without_tasks(role):
        if role == Role.DIAGNOSTICIAN:
            return {"root_causes": ["scope"], "recommendation": "SPLIT"}  # no split_tasks
        return original_payload(role)

    fake._payload = split_without_tasks
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    task = orchestrator.tasks.get_by_key(1, "T001")
    assert task["status"] == "BLOCKED"       # not SKIPPED — nothing replaced it
    assert state == ProjectState.BLOCKED     # so the project cannot complete


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

    class ExpensiveFailingRunner(BaseAgentRunner):
        async def run(self, request):
            calls.append(request.role)
            return AgentResult(status="FAILED", error="boom", cost_usd=5.0)

    monkeypatch.setattr(agent_invoker_module, "create_runner",
                        lambda provider, inference=None: ExpensiveFailingRunner())

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
            (fake.cwd / "marker.txt").write_text("ok\n")  # into the task worktree
        return original_payload(role)

    fake._payload = payload_with_marker
    install_fake(monkeypatch, fake)

    state = await orchestrator.run()

    assert state == ProjectState.COMPLETED
    assert fake.calls.count(Role.DEVELOPER) == 2
    # reviewer only ran once — verification failure short-circuits to Developer
    assert fake.calls.count(Role.REVIEWER) == 1
