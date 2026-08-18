"""Full project run driven by the REAL local OpenAI-compatible provider.

Every other orchestration test substitutes a FakeRunner, so the wiring
between the provider and the parallel harness — gate, host pools, tool
loop, worktree cwd, artifact validation, serialized integration — is
never exercised together. This drives ProjectOrchestrator against a
mocked vLLM endpoint instead: the Developer really calls write_file
through the tool loop, and the change has to reach the integration
branch.
"""

import json
from pathlib import Path

import httpx
import pytest

import harness.agents.openai_compat as oc
from harness.config import HarnessConfig, ProjectConfig, ProviderConfig, VerificationConfig
from harness.git.repository import GitRepository
from harness.orchestrator.project import ProjectOrchestrator
from harness.orchestrator.state_machine import ProjectState

REPO_ROOT = Path(__file__).resolve().parent.parent

PLAN = {
    "summary": "one task",
    "tasks": [{"task_key": "T001", "title": "add feature file",
               "goal": "create feature.txt", "acceptance_criteria": ["exists"],
               "dependencies": []}],
}
BRIEF = {"task": "T001", "summary": "create the file", "files": ["feature.txt"]}
IMPLEMENTATION = {"task": "T001", "summary": "wrote the file",
                  "changed_files": ["feature.txt"]}
TEST_REPORT = {"task": "T001", "summary": "coverage ok"}
REVIEW = {"verdict": "PASS", "summary": "looks right", "blocking_issues": []}


def fenced(payload: dict) -> dict:
    return {"choices": [{"message": {"role": "assistant",
                                     "content": f"```json\n{json.dumps(payload)}\n```"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5}}


@pytest.fixture
def config(tmp_path) -> HarnessConfig:
    workspace = tmp_path / "workspace"
    repo = GitRepository(workspace / "repository")
    repo.init()
    (repo.path / "README.md").write_text("seed\n")
    repo.add_all()
    repo.commit("initial")
    cfg = HarnessConfig(
        project=ProjectConfig(name="e2e", goal="add a feature"),
        workspace_dir=str(workspace),
        # the production default: every role on the local endpoint
        provider=ProviderConfig(type="openai-compatible", model="qwen3.6-27b-fp8"),
        verification=VerificationConfig(language="none", commands=["test -f feature.txt"]),
    )
    cfg.config_path = REPO_ROOT / "config.yaml"
    return cfg


async def test_full_project_runs_through_the_local_provider(config, monkeypatch):
    seen_roles: list[str] = []
    wrote_via_tool = {"done": False}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        messages = body["messages"]
        text = "\n".join(m.get("content") or "" for m in messages)
        already_used_tool = any(m.get("role") == "tool" for m in messages)

        if "project plan JSON" in text:
            seen_roles.append("planner")
            return httpx.Response(200, json=fenced(PLAN))
        if "task brief JSON" in text:
            seen_roles.append("analyst")
            return httpx.Response(200, json=fenced(BRIEF))
        if "implementation JSON" in text:
            if not already_used_tool:
                seen_roles.append("developer:tool")
                return httpx.Response(200, json={
                    "choices": [{"message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "w1", "type": "function", "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"path": "feature.txt",
                                                     "content": "feature\n"})}}]}}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
            wrote_via_tool["done"] = True
            seen_roles.append("developer")
            return httpx.Response(200, json=fenced(IMPLEMENTATION))
        if "test report JSON" in text:
            seen_roles.append("tester")
            return httpx.Response(200, json=fenced(TEST_REPORT))
        if "review JSON" in text:
            seen_roles.append("reviewer")
            return httpx.Response(200, json=fenced(REVIEW))
        raise AssertionError(f"unroutable request: {text[-200:]}")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(oc, "shared_client",
                        lambda base_url, t, k=None: httpx.AsyncClient(
                            base_url=base_url, transport=transport))

    orchestrator = ProjectOrchestrator(config)
    state = await orchestrator.run()

    assert state == ProjectState.COMPLETED, "the project did not finish"
    assert seen_roles == ["planner", "analyst", "developer:tool", "developer",
                          "tester", "reviewer"]

    # the Developer's change was made through the TOOL, inside the task
    # worktree, and reached the integration branch via a serialized merge
    assert wrote_via_tool["done"]
    task = orchestrator.tasks.get_by_key(1, "T001")
    assert task["status"] == "COMPLETED"
    assert task["task_commit"] and task["integration_commit"]
    assert (orchestrator.git.path / "feature.txt").read_text() == "feature\n"

    # deterministic verification really ran against the worktree
    assert orchestrator.runs.has_evaluation(
        orchestrator.db.query_one(
            "SELECT id FROM task_attempts WHERE status = 'PASSED'")["id"],
        "VERIFY_PASS")

    # requests went through the CONFIGURED pool, not a private gate
    assert orchestrator.pools.llm.dispatched_total >= len(seen_roles)
    assert orchestrator.pools.llm.in_flight == 0

    # nothing left behind
    assert orchestrator.worktrees.registered_paths() == []
    assert orchestrator.runs.running_runs() == []
    assert orchestrator.operations.unfinished() == []
    assert orchestrator.events.find_stream_gaps() == []
