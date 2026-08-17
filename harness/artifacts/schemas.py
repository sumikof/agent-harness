"""Structured artifact schemas exchanged between agents.

Agents never continue each other's conversations; these JSON documents
are the only channel between roles (the blackboard).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PlannedTask(BaseModel):
    task_key: str = Field(description="Stable id like T001")
    title: str
    goal: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list, description="task_keys this task depends on")


class ProjectPlan(BaseModel):
    summary: str = ""
    # An empty plan would let the project complete without doing anything;
    # a planner that finds no work must say so in a task, not an empty list.
    tasks: list[PlannedTask] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class TaskBrief(BaseModel):
    task: str
    summary: str = ""
    files: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class Implementation(BaseModel):
    task: str = ""
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)


class TestReport(BaseModel):
    task: str = ""
    summary: str = ""
    tests_added: list[str] = Field(default_factory=list)
    tests_modified: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class BlockingIssue(BaseModel):
    file: str = ""
    issue: str
    required_fix: str = ""


class Review(BaseModel):
    verdict: Literal["PASS", "REPAIR", "REPLAN"]
    score: Optional[float] = None
    summary: str = ""
    blocking_issues: list[BlockingIssue] = Field(default_factory=list)
    non_blocking_notes: list[str] = Field(default_factory=list)
    replan_reason: str = ""


class Diagnosis(BaseModel):
    root_causes: list[str] = Field(default_factory=list)
    wrong_assumptions: list[str] = Field(default_factory=list)
    hidden_dependencies: list[str] = Field(default_factory=list)
    recommendation: Literal["RETRY", "SPLIT", "REPLAN", "BLOCKED"]
    split_tasks: list[PlannedTask] = Field(
        default_factory=list, description="Only when recommendation == SPLIT"
    )
    retry_guidance: str = ""


class VerificationResult(BaseModel):
    """Produced by the deterministic verifier, not by an LLM."""

    passed: bool
    steps: list["VerificationStep"] = Field(default_factory=list)


class VerificationStep(BaseModel):
    command: str
    exit_code: int
    duration_seconds: float
    log_file: str = ""
    tail: str = ""
    output_sha256: str = ""
    output_bytes: int = 0


VerificationResult.model_rebuild()


# -- provenance envelope ----------------------------------------------------


class ArtifactProducer(BaseModel):
    role: str
    agent_run_id: Optional[int] = None


class ArtifactEnvelope(BaseModel):
    """Common metadata wrapper for agent-produced artifacts.

    The domain document lives unchanged under `payload`; the envelope adds
    provenance (who produced it, from which inputs, at which commit) so any
    artifact can be traced back to its producing AgentRun and context.
    """

    schema_version: int = 1
    artifact_type: str
    project_id: Optional[int] = None
    task_id: Optional[int] = None
    attempt_no: Optional[int] = None
    producer: Optional[ArtifactProducer] = None
    base_commit: Optional[str] = None
    input_manifest_hash: Optional[str] = None
    created_at: str = ""
    payload: dict = Field(default_factory=dict)
    evidence: list[dict] = Field(default_factory=list)


# -- large output retention -------------------------------------------------


class SpilledOutput(BaseModel):
    """Bounded preview of an oversized output whose full text was spilled
    to an artifact file. Nothing is lost: the artifact holds every byte."""

    artifact_path: str
    sha256: str
    total_bytes: int
    truncated: bool
    head: str = ""
    tail: str = ""

    def render(self) -> str:
        if not self.truncated:
            return self.head
        return (
            f"{self.head}\n"
            f"... (output truncated: {self.total_bytes} bytes total, "
            f"full copy at {self.artifact_path}, sha256 {self.sha256[:16]}...)\n"
            f"{self.tail}"
        )
