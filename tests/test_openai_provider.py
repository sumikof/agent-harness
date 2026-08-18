"""Local OpenAI-compatible provider: tool loop, permissions, retries,
semaphore, stable schemas (spec items 17, 22-23, 42, 55, 65, 67, 71)."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import harness.agents.openai_compat as oc
from harness.agents.local_tools import (
    LocalToolExecutor,
    decide_local_tool_use,
    tool_schema_hash,
    tool_schemas_for,
)
from harness.agents.openai_compat import LocalOpenAICompatibleAgentRunner
from harness.agents.profile import RepeatGuardConfig, ResolvedAgentRunSpec
from harness.config import InferenceConfig
from harness.orchestrator.resources import PrefixAffinityGate
from harness.orchestrator.state_machine import Role

# ---------------------------------------------------------------------------
# tool schema stability + permissions
# ---------------------------------------------------------------------------


def test_tool_schemas_fixed_order_and_stable_hash():
    a = tool_schemas_for(Role.DEVELOPER)
    b = tool_schemas_for(Role.DEVELOPER)
    assert [t["function"]["name"] for t in a] == [t["function"]["name"] for t in b]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert tool_schema_hash(Role.DEVELOPER) == tool_schema_hash(Role.DEVELOPER)
    # roles differ where their catalogs differ
    assert tool_schema_hash(Role.DEVELOPER) != tool_schema_hash(Role.REVIEWER)
    reviewer_names = [t["function"]["name"] for t in tool_schemas_for(Role.REVIEWER)]
    assert "write_file" not in reviewer_names and "edit_file" not in reviewer_names


def test_local_tool_permissions(tmp_path):
    allowed, _ = decide_local_tool_use(Role.REVIEWER, "write_file",
                                       {"path": "src/x.py"}, tmp_path)
    assert not allowed
    allowed, _ = decide_local_tool_use(Role.TESTER, "write_file",
                                       {"path": "src/x.py"}, tmp_path)
    assert not allowed  # tester writes tests only
    allowed, _ = decide_local_tool_use(Role.TESTER, "write_file",
                                       {"path": "tests/test_x.py"}, tmp_path)
    assert allowed
    allowed, reason = decide_local_tool_use(Role.DEVELOPER, "run_command",
                                            {"command": "git commit -m x"}, tmp_path)
    assert not allowed and "harness" in reason
    allowed, _ = decide_local_tool_use(Role.REVIEWER, "run_command",
                                       {"command": "echo hi > out.txt"}, tmp_path)
    assert not allowed  # read-only shell for non-developers
    allowed, _ = decide_local_tool_use(Role.DEVELOPER, "write_file",
                                       {"path": "../outside.txt"}, tmp_path)
    assert not allowed


async def test_executor_denies_and_reports_as_text(tmp_path):
    executor = LocalToolExecutor(role=Role.REVIEWER, cwd=tmp_path, repo_root=tmp_path)
    out = await executor.execute("write_file", {"path": "a.txt", "content": "x"})
    assert out.startswith("TOOL DENIED")
    (tmp_path / "f.txt").write_text("line1\nline2\n")
    out = await executor.execute("read_file", {"path": "f.txt"})
    assert "line1" in out


# ---------------------------------------------------------------------------
# runner behavior against a fake endpoint
# ---------------------------------------------------------------------------


def make_spec(tmp_path, role=Role.ANALYST, **overrides) -> ResolvedAgentRunSpec:
    values = dict(
        provider="openai-compatible", model="qwen3.6-27b-fp8", role=role.value,
        profile_id="x", profile_version="1", profile_hash="h",
        max_turns=8, cwd=str(tmp_path), repo_root=str(tmp_path),
        base_url="http://fake/v1", sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        repeat_guard=RepeatGuardConfig(enabled=True, warn_after=2, abort_after=3),
        system_prompt="system", prompt="user prompt",
    )
    values.update(overrides)
    return ResolvedAgentRunSpec(**values)


def install_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def fake_shared_client(base_url, timeout_seconds, api_key=None):
        return httpx.AsyncClient(
            base_url=base_url, transport=transport,
            headers=oc.auth_headers(api_key),
        )

    monkeypatch.setattr(oc, "shared_client", fake_shared_client)


def chat_response(content=None, tool_calls=None, usage=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={
        "choices": [{"message": message}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    })


async def test_tool_loop_executes_and_finishes(tmp_path, monkeypatch):
    (tmp_path / "hello.txt").write_text("hello world\n")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return chat_response(tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "read_file",
                             "arguments": json.dumps({"path": "hello.txt"})},
            }])
        # second turn: the tool result must be in the message history
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        assert tool_msgs and "hello world" in tool_msgs[0]["content"]
        return chat_response(content='```json\n{"ok": true}\n```')

    install_transport(monkeypatch, handler)
    runner = LocalOpenAICompatibleAgentRunner(InferenceConfig(), PrefixAffinityGate(4))
    result = await runner.run(make_spec(tmp_path))
    assert result.status == "COMPLETED"
    assert result.structured_output == {"ok": True}
    assert result.token_usage["input_tokens"] == 20
    assert result.telemetry["tool_calls"] == 1
    # sampling profile reached the wire, unchanged per request
    assert calls[0]["temperature"] == 0.6 and calls[0]["top_k"] == 20
    assert calls[0]["stream"] is False


async def test_transient_429_retried_in_session(tmp_path, monkeypatch):
    """Overload (429) is backpressure: retried inside the SAME session with
    history intact — never surfaced as a task failure or extra attempt."""
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) <= 2:
            return httpx.Response(429, text="overloaded")
        return chat_response(content="done")

    install_transport(monkeypatch, handler)
    inference = InferenceConfig(transient_retries=4, transient_retry_base_delay=0.0)
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(4))
    result = await runner.run(make_spec(tmp_path))
    assert result.status == "COMPLETED"
    assert len(attempts) == 3


async def test_permanent_error_not_retried(tmp_path, monkeypatch):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, text="unauthorized")

    install_transport(monkeypatch, handler)
    inference = InferenceConfig(transient_retries=4, transient_retry_base_delay=0.0)
    runner = LocalOpenAICompatibleAgentRunner(inference, PrefixAffinityGate(4))
    result = await runner.run(make_spec(tmp_path))
    assert result.status == "FAILED"
    assert result.failure_kind is not None and result.failure_kind.value == "PERMANENT"
    assert len(attempts) == 1


async def test_llm_semaphore_caps_concurrent_requests(tmp_path, monkeypatch):
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return chat_response(content="ok")

    install_transport(monkeypatch, handler)
    gate = PrefixAffinityGate(3)
    runner = LocalOpenAICompatibleAgentRunner(InferenceConfig(), gate)
    specs = [make_spec(tmp_path) for _ in range(12)]
    results = await asyncio.gather(*(runner.run(s) for s in specs))
    assert all(r.status == "COMPLETED" for r in results)
    assert peak <= 3


async def test_repeat_guard_aborts_looping_session(tmp_path, monkeypatch):
    (tmp_path / "same.txt").write_text("x\n")

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response(tool_calls=[{
            "id": "c", "type": "function",
            "function": {"name": "read_file",
                         "arguments": json.dumps({"path": "same.txt"})},
        }])

    install_transport(monkeypatch, handler)
    runner = LocalOpenAICompatibleAgentRunner(InferenceConfig(), PrefixAffinityGate(4))
    result = await runner.run(make_spec(tmp_path, max_turns=20))
    assert result.status == "FAILED"
    assert result.loop_detected
    assert "LOOP_DETECTED" in (result.error or "")


async def test_reasoning_content_never_forwarded(tmp_path, monkeypatch):
    """Hidden reasoning is used within a turn only — it must not be appended
    back into the message history (Fresh Agent + Structured Handoff)."""
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            response = chat_response(tool_calls=[{
                "id": "c", "type": "function",
                "function": {"name": "list_directory",
                             "arguments": json.dumps({"path": "."})},
            }])
            data = json.loads(response.content)
            data["choices"][0]["message"]["reasoning_content"] = "SECRET-CHAIN"
            return httpx.Response(200, json=data)
        return chat_response(content="done")

    install_transport(monkeypatch, handler)
    runner = LocalOpenAICompatibleAgentRunner(InferenceConfig(), PrefixAffinityGate(4))
    result = await runner.run(make_spec(tmp_path))
    assert result.status == "COMPLETED"
    assert "SECRET-CHAIN" not in json.dumps(bodies[1]["messages"])
    assert "SECRET-CHAIN" not in result.output_text


def test_capabilities_meet_role_requirements():
    from harness.agents.profile import ROLE_REQUIRED_CAPABILITIES

    caps = LocalOpenAICompatibleAgentRunner(InferenceConfig()).capabilities()
    for role, required in ROLE_REQUIRED_CAPABILITIES.items():
        assert caps.missing(required) == [], f"{role} requirements unmet"
