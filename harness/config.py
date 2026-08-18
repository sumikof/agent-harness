"""Configuration loading and validation."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str
    repository: str = "repository"
    base_branch: str = "main"
    goal: str = ""


class RoleProviderConfig(BaseModel):
    type: Optional[str] = None
    model: Optional[str] = None


class ProviderConfig(BaseModel):
    # Default execution engine: the local OpenAI-compatible adapter serving
    # Qwen3.6-27B-FP8 via vLLM on DGX Spark. The provider abstraction stays:
    # "claude" (Claude Agent SDK) and future engines remain selectable
    # globally or per role.
    type: str = "openai-compatible"
    model: str = "qwen3.6-27b-fp8"
    roles: dict[str, RoleProviderConfig] = Field(default_factory=dict)

    def for_role(self, role: str) -> tuple[str, str]:
        """Return (provider_type, model) for a role, falling back to defaults."""
        override = self.roles.get(role)
        if override is None:
            return self.type, self.model
        return override.type or self.type, override.model or self.model


class SamplingConfig(BaseModel):
    """Generation profile for the local model. Fixed per profile — never
    varied per request, so identical contexts produce identical requests."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


class InferenceConcurrencyConfig(BaseModel):
    # Process-wide cap on in-flight LLM HTTP requests (semaphore). Matches
    # the serving-side --max-num-seqs baseline.
    max_requests: int = 16


class InferenceConfig(BaseModel):
    """Local OpenAI-compatible serving endpoint (vLLM on DGX Spark)."""

    provider: str = "openai-compatible"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "not-needed"  # vLLM ignores it; the SDK requires a value
    model: str = "qwen3.6-27b-fp8"
    # Context profile selects max input+output budget. `performance` is the
    # production default; longer profiles are opt-in per task, never global.
    context_profile: str = "performance"
    context_profiles: dict[str, int] = Field(
        default_factory=lambda: {
            "performance": 65536,
            "long": 131072,
            "maximum": 262144,
        }
    )
    # Input token budget within the context profile: the rest is reserved
    # for reasoning, tool calls, and model output. Approximate (chars/4).
    input_budget_tokens: int = 50000
    # Reserved output tokens per request.
    max_output_tokens: int = 8192
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    # Per-role sampling overrides (partial; unset fields fall back).
    role_sampling: dict[str, SamplingConfig] = Field(default_factory=dict)
    concurrency: InferenceConcurrencyConfig = Field(
        default_factory=InferenceConcurrencyConfig
    )
    # Internal agents don't need token-by-token display; non-streaming
    # reduces host CPU / HTTP overhead. Benchmarkable.
    streaming: bool = False
    request_timeout_seconds: int = 600
    # In-loop transient retry (429 / 5xx / transport) inside one session:
    # message history is preserved, the failed HTTP call is repeated.
    # Bounded; never consumes a task attempt.
    transient_retries: int = 4
    transient_retry_base_delay: float = 2.0

    def max_model_len(self) -> int:
        return self.context_profiles.get(self.context_profile, 65536)

    def sampling_for_role(self, role: str) -> SamplingConfig:
        override = self.role_sampling.get(role)
        if override is None:
            return self.sampling
        return override


class ResourcePoolsConfig(BaseModel):
    """Named semaphores separating LLM inference slots from host-heavy
    work (builds/tests) and the strictly-serialized git integration."""

    llm: int = 16
    heavy_build: int = 2
    heavy_test: int = 2
    git_integration: int = 1


class ParallelismConfig(BaseModel):
    max_parallel_tasks: int = 16
    # Upper bound on concurrently RUNNING AgentRuns (DB invariant). Defaults
    # to max_parallel_tasks: each task runs one role at a time.
    max_parallel_agent_runs: Optional[int] = None
    resource_pools: ResourcePoolsConfig = Field(default_factory=ResourcePoolsConfig)
    # Scheduling-fairness: a READY task skipped this many scheduling rounds
    # is dispatched next regardless of prefix affinity.
    starvation_rounds: int = 8

    def agent_run_limit(self) -> int:
        return self.max_parallel_agent_runs or self.max_parallel_tasks


class GitStrategyConfig(BaseModel):
    # Parallel tasks each get an isolated worktree + branch; integration
    # into the base branch is strictly serialized.
    task_worktrees: bool = True
    integration_strategy: str = "serialized"
    # Directory (relative to workspace) holding task worktrees.
    worktrees_dir: str = "worktrees"
    branch_prefix: str = "harness/task"


class DatabaseConfig(BaseModel):
    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5000
    # All writes go through one serialized writer (process-wide lock).
    # Parallel agent coroutines/threads never race write transactions.
    single_writer: bool = True


class VerificationConfig(BaseModel):
    language: str = "none"
    commands: list[str] = Field(default_factory=list)
    timeout_seconds: int = 900


class BudgetConfig(BaseModel):
    project_usd: float = 100.0
    task_usd: float = 10.0
    agent_run_usd: float = 3.0


class LimitsConfig(BaseModel):
    max_attempts: int = 3
    max_turns: int = 60
    max_agent_runs_per_task: int = 25
    max_execution_seconds: int = 172800
    # One agent session (distinct from provider request / verification /
    # tool-call timeouts, which are configured where they apply).
    agent_run_timeout_seconds: int = 3600
    # Outputs larger than this are spilled to an artifact file and only a
    # bounded preview enters agent context.
    max_inline_output_chars: int = 30000
    # Fresh repair attempts allowed after an integration conflict before
    # the task is BLOCKED.
    max_integration_repairs: int = 2


class RepeatGuardSettings(BaseModel):
    """Repeat Action Guard (loop detection); enforced only on providers
    that support a pre-tool hook."""

    enabled: bool = True
    warn_after: int = 3
    abort_after: int = 5
    # Tools whose legitimate repetition (e.g. read-only polling) should not
    # trip the guard.
    exempt_tools: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class HarnessConfig(BaseModel):
    project: ProjectConfig
    workspace_dir: str = "workspace"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    git: GitStrategyConfig = Field(default_factory=GitStrategyConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    repeat_guard: RepeatGuardSettings = Field(default_factory=RepeatGuardSettings)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Resolved at load time; not part of the YAML schema.
    config_path: Optional[Path] = None

    @property
    def workspace_path(self) -> Path:
        base = self.config_path.parent if self.config_path else Path.cwd()
        ws = Path(self.workspace_dir)
        return ws if ws.is_absolute() else base / ws

    @property
    def repository_path(self) -> Path:
        repo = Path(self.project.repository)
        return repo if repo.is_absolute() else self.workspace_path / repo

    @property
    def worktrees_path(self) -> Path:
        wd = Path(self.git.worktrees_dir)
        return wd if wd.is_absolute() else self.workspace_path / wd

    @property
    def db_path(self) -> Path:
        return self.workspace_path / "harness.db"

    @property
    def artifacts_path(self) -> Path:
        return self.workspace_path / "artifacts"

    @property
    def logs_path(self) -> Path:
        return self.workspace_path / "logs"

    @property
    def prompts_path(self) -> Path:
        """A `prompts/` directory next to the config overrides the packaged
        role prompts; otherwise the ones shipped inside the package are used,
        so installed wheels work without a source checkout."""
        base = self.config_path.parent if self.config_path else Path.cwd()
        local = base / "prompts"
        if local.is_dir():
            return local
        return Path(str(importlib.resources.files("harness").joinpath("prompts")))


def load_config(path: str | Path) -> HarnessConfig:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = HarnessConfig(**raw)
    config.config_path = path
    return config
