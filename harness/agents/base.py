"""Agent provider abstraction.

Everything Claude-Agent-SDK-specific lives inside ClaudeAgentRunner, so
the harness can later swap in other engines (Codex, local LLMs, ...) per
role via config. The rest of the harness only sees AgentRunner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Type

from pydantic import BaseModel, ValidationError

from ..orchestrator.state_machine import Role
from ..security.hooks import build_pretooluse_hook
from ..security.permissions import allowed_tools_for


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
    resume_session_id: Optional[str] = None


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


class AgentRunner(Protocol):
    async def run(self, request: AgentRequest) -> AgentResult: ...


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


# -- Claude Agent SDK adapter ----------------------------------------------


class ClaudeAgentRunner:
    """Runs one fresh agent session via the Claude Agent SDK."""

    async def run(self, request: AgentRequest) -> AgentResult:
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
            )

        hook = build_pretooluse_hook(request.role, request.repo_root)
        options = ClaudeAgentOptions(
            system_prompt=request.system_prompt,
            model=request.model,
            cwd=str(request.cwd),
            allowed_tools=allowed_tools_for(request.role),
            permission_mode="acceptEdits",
            max_turns=request.max_turns,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[hook])]},
        )
        if request.resume_session_id:
            options.resume = request.resume_session_id

        text_parts: list[str] = []
        result = AgentResult(status="FAILED")
        try:
            async for message in query(prompt=request.prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
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
        except Exception as exc:  # SDK/transport failure — recoverable via retry
            result.status = "FAILED"
            result.error = f"{type(exc).__name__}: {exc}"

        result.output_text = "\n".join(text_parts)
        if result.status == "COMPLETED":
            result.structured_output = extract_json(result.output_text)
        return result


def create_runner(provider_type: str) -> AgentRunner:
    if provider_type == "claude":
        return ClaudeAgentRunner()
    raise ValueError(f"unknown agent provider: {provider_type}")
