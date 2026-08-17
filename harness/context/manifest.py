"""ContextManifest: a durable record of exactly what one AgentRun received.

Before an agent is dispatched, every harness-visible input section
(project / task / attempt context, extra material, correction feedback)
is written to an artifact file and hashed; the manifest ties the refs
together with the prompt template hash and agent profile hash. From the
manifest alone the full input of any past AgentRun can be reconstructed.

The manifest MUST be persisted before dispatch — an agent never starts
without one.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ContextRef(BaseModel):
    """Pointer to one persisted context section."""

    artifact: str  # file path of the stored section text
    sha256: str


class ContextManifest(BaseModel):
    schema_version: int = 1
    project_id: int
    task_id: Optional[int] = None
    attempt_no: Optional[int] = None
    role: str
    base_commit: Optional[str] = None
    sections: dict[str, ContextRef] = Field(default_factory=dict)
    prompt_template_hash: str = ""
    agent_profile_hash: str = ""
    prompt_sha256: str = ""      # hash of the fully assembled user prompt
    system_prompt_sha256: str = ""
    created_at: str = ""
