"""Agent profiles, provider capabilities, and resolved run specs.

AgentProfile bundles everything that defines *how* a role runs (provider,
model, prompt, permissions, limits) into one versioned, hashable unit, so
every AgentRun records exactly which profile produced it.

AgentCapabilities makes provider feature differences explicit: a role
declares what it requires, and the harness refuses to dispatch on a
provider that cannot deliver — no silent degraded mode.

ResolvedAgentRunSpec freezes, before dispatch, everything an AgentRun
will actually execute with. It is persisted to durable storage first;
`persistable_dump()` replaces the full prompt bodies with hashes (the
bodies themselves are reconstructable via the ContextManifest).
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from pydantic import BaseModel, Field

from ..orchestrator.state_machine import Role

PROFILE_SCHEMA_VERSION = 1
PROFILE_VERSION = "1"


class AgentCapabilities(BaseModel):
    structured_output: bool = False
    session_resume: bool = False
    tool_permissions: bool = False
    pre_tool_hook: bool = False
    post_tool_hook: bool = False
    cancellation: bool = False
    hard_termination: bool = False
    usage_reporting: bool = False

    @classmethod
    def all_supported(cls) -> "AgentCapabilities":
        return cls(**{name: True for name in cls.model_fields})

    def missing(self, required: list[str]) -> list[str]:
        return [name for name in required if not getattr(self, name, False)]


# Every role relies on structured output for its artifact, and on tool
# permissions + the PreToolUse hook for enforced (not just prompted)
# read-only / write-path restrictions.
ROLE_REQUIRED_CAPABILITIES: dict[Role, list[str]] = {
    role: ["structured_output", "tool_permissions", "pre_tool_hook"] for role in Role
}


class RepeatGuardConfig(BaseModel):
    """Repeat Action Guard thresholds; only enforced on providers with a
    pre_tool_hook capability."""

    enabled: bool = True
    warn_after: int = 3
    abort_after: int = 5
    exempt_tools: list[str] = Field(default_factory=list)


class AgentProfile(BaseModel):
    schema_version: int = PROFILE_SCHEMA_VERSION
    profile_id: str          # e.g. "developer@claude"
    profile_version: str = PROFILE_VERSION
    role: str
    provider: str
    model: str
    prompt_file: str
    prompt_template_hash: str
    required_capabilities: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)
    output_schema: str = ""  # pydantic model name of the structured artifact
    max_turns: int = 60
    timeout_seconds: int = 3600
    budget_usd: float = 0.0

    def profile_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResolvedAgentRunSpec(BaseModel):
    schema_version: int = 1
    provider: str
    model: str
    role: str
    profile_id: str
    profile_version: str
    profile_hash: str
    prompt_template_hash: str = ""
    context_manifest_path: str = ""
    context_manifest_hash: str = ""
    output_schema: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)
    max_turns: int = 60
    timeout_seconds: Optional[int] = None
    budget_usd: Optional[float] = None
    resume_session_id: Optional[str] = None
    cwd: str = ""
    repo_root: str = ""
    repeat_guard: Optional[RepeatGuardConfig] = None
    # Local OpenAI-compatible provider extras (unused by other providers).
    base_url: str = ""
    sampling: dict = Field(default_factory=dict)
    max_output_tokens: Optional[int] = None
    tool_schema_hash: str = ""
    prefix_group_key: str = ""
    # Full texts needed at execution time; excluded from the persisted dump.
    system_prompt: str = ""
    prompt: str = ""

    def persistable_dump(self) -> dict:
        """Spec as stored in durable storage: prompt bodies become hashes
        (they are reconstructable from the ContextManifest artifacts)."""
        data = self.model_dump(mode="json", exclude={"system_prompt", "prompt"})
        data["system_prompt_sha256"] = _sha256(self.system_prompt)
        data["prompt_sha256"] = _sha256(self.prompt)
        return data


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
