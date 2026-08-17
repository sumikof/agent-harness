"""Agent provider abstraction.

Everything Claude-Agent-SDK-specific lives inside ClaudeAgentRunner, so
the harness can later swap in other engines (Codex, local LLMs, ...) per
role via config. The rest of the harness only sees AgentRunner.

The runner contract is split into three phases:

    capabilities()        what this provider can actually do
    resolve(request)      freeze everything the run will execute with
    run(spec)             execute the frozen spec

The resolved spec is persisted by the invoker BEFORE dispatch, so a
crashed run can always be explained after the fact.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional, Protocol, Type

from pydantic import BaseModel, ValidationError

from ..orchestrator.state_machine import Role
from ..security.hooks import RepeatActionGuard, build_pretooluse_hook
from ..security.permissions import TEST_WRITE_GLOBS, allowed_tools_for
from .profile import (
    AgentCapabilities,
    AgentProfile,
    RepeatGuardConfig,
    ResolvedAgentRunSpec,
)


class FailureKind(StrEnum):
    """Provider-failure classification driving the retry layers.

    TRANSIENT failures may be retried at the provider layer (bounded,
    with backoff); PERMANENT ones (auth, config, capability) must never
    be, and surface immediately.
    """

    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"


@dataclass
class RoleSpec:
    """Static definition of one agent role."""

    role: Role
    prompt_file: str                       # under prompts/
    output_model: Optional[Type[BaseModel]]
    output_artifact: str                   # filename saved under artifacts/tasks/<key>/
    mutates_repo: bool = False


@dataclass
class AgentRequest:
    role: Role
    system_prompt: str
    prompt: str
    cwd: Path
    repo_root: Path
    model: str
    max_turns: int = 60
    timeout_seconds: Optional[int] = None
    resume_session_id: Optional[str] = None
    profile: Optional[AgentProfile] = None
    context_manifest_path: str = ""
    context_manifest_hash: str = ""
    repeat_guard: Optional[RepeatGuardConfig] = None


@dataclass
class AgentResult:
    status: str                            # COMPLETED | FAILED
    output_text: str = ""
    structured_output: Optional[dict] = None
    session_id: Optional[str] = None
    token_usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    num_turns: int = 0
    error: Optional[str] = None
    failure_kind: Optional[FailureKind] = None
    loop_detected: bool = False
    loop_warnings: list[str] = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)


class AgentRunner(Protocol):
    def capabilities(self) -> AgentCapabilities: ...
    async def resolve(self, request: AgentRequest) -> ResolvedAgentRunSpec: ...
    async def run(self, spec: ResolvedAgentRunSpec) -> AgentResult: ...


# -- provider error classification ------------------------------------------

_PERMANENT_MARKERS = (
    "401", "403", "authentication", "unauthorized", "permission denied",
    "invalid_api_key", "api key", "billing", "not_found_error", "model not found",
    "invalid model", "is not installed",
)
_TRANSIENT_MARKERS = (
    "429", "rate limit", "overloaded", "timeout", "timed out", "connection",
    "temporarily", "500", "502", "503", "504", "server error", "socket",
    "reset by peer", "unavailable",
)


def classify_provider_error(message: str) -> FailureKind:
    """Sort a provider failure into the retryable / non-retryable bucket.

    Unknown failures default to TRANSIENT: a bounded retry of an unknown
    error is cheap, while refusing to retry a flaky transport is not.
    """
    text = (message or "").lower()
    for marker in _PERMANENT_MARKERS:
        if marker in text:
            return FailureKind.PERMANENT
    for marker in _TRANSIENT_MARKERS:
        if marker in text:
            return FailureKind.TRANSIENT
    return FailureKind.TRANSIENT


# -- structured output ------------------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """Pull the last JSON object out of agent output.

    Tries fenced ```json blocks first (last one wins), then the largest
    top-level {...} span in the raw text.
    """
    for match in reversed(_JSON_FENCE.findall(text)):
        try:
            data = json.loads(match)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    while start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def validate_output(
    result: AgentResult, model_type: Type[BaseModel]
) -> tuple[Optional[BaseModel], Optional[str]]:
    """Validate an agent's structured output. Returns (model, error)."""
    data = result.structured_output or extract_json(result.output_text)
    if data is None:
        return None, "agent produced no parseable JSON output"
    try:
        return model_type.model_validate(data), None
    except ValidationError as exc:
        return None, f"agent output failed schema validation: {exc}"


# -- shared resolve logic ---------------------------------------------------


class BaseAgentRunner:
    """Provider-independent resolve(): freeze the request + profile into a
    ResolvedAgentRunSpec. Providers override capabilities() honestly and
    implement run()."""

    provider_name = "base"

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities.all_supported()

    async def resolve(self, request: AgentRequest) -> ResolvedAgentRunSpec:
        profile = request.profile
        caps = self.capabilities()
        guard_config = request.repeat_guard
        if guard_config is not None and not caps.pre_tool_hook:
            guard_config = None  # capability-gated: never fake enforcement
        resume = request.resume_session_id if caps.session_resume else None
        writable = []
        if request.role == Role.DEVELOPER:
            writable = ["**"]
        elif request.role == Role.TESTER:
            writable = list(TEST_WRITE_GLOBS)
        return ResolvedAgentRunSpec(
            provider=profile.provider if profile else self.provider_name,
            model=request.model,
            role=request.role.value,
            profile_id=profile.profile_id if profile else f"{request.role.value}@{self.provider_name}",
            profile_version=profile.profile_version if profile else "unversioned",
            profile_hash=profile.profile_hash() if profile else "",
            prompt_template_hash=profile.prompt_template_hash if profile else "",
            context_manifest_path=request.context_manifest_path,
            context_manifest_hash=request.context_manifest_hash,
            output_schema=profile.output_schema if profile else "",
            allowed_tools=allowed_tools_for(request.role),
            writable_paths=writable,
            max_turns=request.max_turns,
            timeout_seconds=request.timeout_seconds,
            budget_usd=profile.budget_usd if profile else None,
            resume_session_id=resume,
            cwd=str(request.cwd),
            repo_root=str(request.repo_root),
            repeat_guard=guard_config,
            system_prompt=request.system_prompt,
            prompt=request.prompt,
        )


# -- Claude Agent SDK adapter ----------------------------------------------


class ClaudeAgentRunner(BaseAgentRunner):
    """Runs one fresh agent session via the Claude Agent SDK."""

    provider_name = "claude"

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            structured_output=True,   # via JSON extraction from final output
            session_resume=True,
            tool_permissions=True,
            pre_tool_hook=True,
            post_tool_hook=False,
            cancellation=True,        # cooperative: cancelling the query task
            hard_termination=False,
            usage_reporting=True,
        )

    async def run(self, spec: ResolvedAgentRunSpec) -> AgentResult:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                HookMatcher,
                ResultMessage,
                TextBlock,
                query,
            )
        except ImportError as exc:
            return AgentResult(
                status="FAILED",
                error=f"claude-agent-sdk is not installed: {exc}. pip install 'agent-harness[claude]'",
                failure_kind=FailureKind.PERMANENT,
            )

        role = Role(spec.role)
        guard: Optional[RepeatActionGuard] = None
        if spec.repeat_guard is not None and spec.repeat_guard.enabled:
            guard = RepeatActionGuard(
                warn_after=spec.repeat_guard.warn_after,
                abort_after=spec.repeat_guard.abort_after,
                exempt_tools=spec.repeat_guard.exempt_tools,
            )
        hook = build_pretooluse_hook(role, Path(spec.repo_root), guard)
        options = ClaudeAgentOptions(
            system_prompt=spec.system_prompt,
            model=spec.model,
            cwd=spec.cwd,
            allowed_tools=list(spec.allowed_tools),
            permission_mode="acceptEdits",
            max_turns=spec.max_turns,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[hook])]},
        )
        if spec.resume_session_id:
            options.resume = spec.resume_session_id

        text_parts: list[str] = []
        tool_calls = 0
        result = AgentResult(status="FAILED")

        async def consume() -> None:
            nonlocal tool_calls
            async for message in query(prompt=spec.prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif type(block).__name__ == "ToolUseBlock":
                            tool_calls += 1
                elif isinstance(message, ResultMessage):
                    result.session_id = getattr(message, "session_id", None)
                    result.cost_usd = getattr(message, "total_cost_usd", None) or 0.0
                    result.num_turns = getattr(message, "num_turns", 0) or 0
                    usage = getattr(message, "usage", None) or {}
                    result.token_usage = usage if isinstance(usage, dict) else {}
                    is_error = getattr(message, "is_error", False)
                    result.status = "FAILED" if is_error else "COMPLETED"
                    if is_error:
                        result.error = str(getattr(message, "result", "agent reported error"))

        try:
            if spec.timeout_seconds:
                # Cooperative cancellation: the query task is cancelled at
                # the deadline; the SDK cleans up its child process.
                await asyncio.wait_for(consume(), timeout=spec.timeout_seconds)
            else:
                await consume()
        except asyncio.TimeoutError:
            result.status = "FAILED"
            result.error = f"agent run exceeded timeout of {spec.timeout_seconds}s"
            result.failure_kind = FailureKind.TRANSIENT
        except Exception as exc:  # SDK/transport failure — classified for retry
            result.status = "FAILED"
            result.error = f"{type(exc).__name__}: {exc}"
            result.failure_kind = classify_provider_error(result.error)

        result.output_text = "\n".join(text_parts)
        if result.status == "COMPLETED":
            result.structured_output = extract_json(result.output_text)
        elif result.failure_kind is None and result.error:
            result.failure_kind = classify_provider_error(result.error)
        if guard is not None:
            result.loop_detected = guard.loop_detected
            result.loop_warnings = list(guard.warnings)
        result.telemetry = {"tool_calls": tool_calls}
        return result


def create_runner(provider_type: str) -> AgentRunner:
    if provider_type == "claude":
        return ClaudeAgentRunner()
    raise ValueError(f"unknown agent provider: {provider_type}")
