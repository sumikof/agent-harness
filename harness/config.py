"""Configuration loading and validation."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# Below this the context profile cannot hold a usable prompt at all.
MIN_INPUT_HEADROOM_TOKENS = 1024
# Room kept free for tool-loop growth per prompt: one MAX_TOOL_OUTPUT_CHARS
# tool result (~7.5k tokens at 4 chars/token) plus the assistant turn.
TOOL_LOOP_RESERVE_TOKENS = 8192


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
    model_config = ConfigDict(extra="forbid")

    # Process-wide cap on in-flight LLM HTTP requests (semaphore). Matches
    # the serving-side --max-num-seqs baseline.
    max_requests: int = Field(default=16, gt=0)


class InferenceConfig(BaseModel):
    """Local OpenAI-compatible serving endpoint (vLLM on DGX Spark).

    There is deliberately no `provider` field here: `provider.type` (and
    the per-role overrides) decide which engine a role dispatches to, and a
    second copy of that choice would either be ignored or contradict the
    authoritative one. Unknown keys are rejected rather than ignored, so a
    stale or misspelled setting fails at load instead of silently doing
    nothing.
    """

    model_config = ConfigDict(extra="forbid")

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
    input_budget_tokens: int = Field(default=50000, gt=0)
    # Reserved output tokens per request. Must be positive: the value is
    # forwarded as `max_tokens`, and zero or negative makes the server
    # reject every request — while leaving MORE apparent input headroom,
    # so a bare headroom check would accept it.
    max_output_tokens: int = Field(default=8192, gt=0)
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

    @model_validator(mode="after")
    def _profile_fits(self) -> "InferenceConfig":
        # Checked on the whole model: `context_profiles` is declared after
        # `context_profile`, so a field validator would not see it yet.
        if self.context_profile not in self.context_profiles:
            raise ValueError(
                f"unknown context_profile '{self.context_profile}'; "
                f"available: {sorted(self.context_profiles)}"
            )
        # An output reservation that consumes the whole window leaves no room
        # for input. Clamping to a one-token budget would hide that: every
        # request would still reserve more than the server can hold.
        if self.input_headroom() < MIN_INPUT_HEADROOM_TOKENS:
            raise ValueError(
                f"max_output_tokens={self.max_output_tokens} leaves only "
                f"{self.input_headroom()} input tokens in context_profile "
                f"'{self.context_profile}' (window {self.max_model_len()}); "
                f"at least {MIN_INPUT_HEADROOM_TOKENS} are required"
            )
        return self

    def max_model_len(self) -> int:
        return self.context_profiles.get(self.context_profile, 65536)

    def effective_input_budget(self) -> int:
        """Input token budget that the selected context profile can hold.

        `context_profile` has to change behaviour, not just documentation:
        the budget can never exceed what is left of the profile's window
        after the reserved output. A configured `input_budget_tokens` that
        does not fit is clamped rather than silently overrunning the
        server's --max-model-len. The reservation itself is validated at
        load, so this never has to invent a degenerate budget.
        """
        return min(self.input_budget_tokens, self.input_headroom())

    def input_headroom(self) -> int:
        return self.max_model_len() - self.max_output_tokens

    def prompt_budget_tokens(self) -> int:
        """Budget for the INITIAL prompt: the input budget minus room for at
        least one tool turn (assistant message + one full tool result).

        Building the first prompt right up to the input budget means the
        very first tool call pushes the history over it, and the only thing
        left to elide is the result the model has not seen yet. The
        reserve keeps one whole turn inside the budget so elision always
        has an already-seen turn to take space from first.
        """
        effective = self.effective_input_budget()
        return effective - min(TOOL_LOOP_RESERVE_TOKENS, effective // 4)

    def sampling_for_role(self, role: str) -> SamplingConfig:
        """The global profile with the role's EXPLICIT overrides applied.

        A role entry is a partial override: only the fields actually
        present in the configuration win. Returning the parsed override
        directly would let pydantic's class defaults silently overwrite a
        customized global profile (e.g. a global top_p alongside a
        reviewer that only sets temperature).
        """
        override = self.role_sampling.get(role)
        if override is None:
            return self.sampling
        explicit = override.model_dump(exclude_unset=True)
        return self.sampling.model_copy(update=explicit)


class ResourcePoolsConfig(BaseModel):
    """Named semaphores separating LLM inference slots from host-heavy
    work (builds/tests) and the strictly-serialized git integration.

    Every size must be positive: a pool of 0 is a semaphore nothing can
    ever acquire, so (for example) `heavy_test: 0` would hang every task
    forever the moment deterministic verification asks for a slot —
    silently, with no error to point at.
    """

    llm: int = Field(default=16, gt=0)
    heavy_build: int = Field(default=2, gt=0)
    heavy_test: int = Field(default=2, gt=0)
    git_integration: int = Field(default=1, gt=0)


class ParallelismConfig(BaseModel):
    max_parallel_tasks: int = Field(default=16, gt=0)
    # Upper bound on concurrently RUNNING AgentRuns (DB invariant). Defaults
    # to max_parallel_tasks: each task runs one role at a time.
    max_parallel_agent_runs: Optional[int] = Field(default=None, gt=0)
    resource_pools: ResourcePoolsConfig = Field(default_factory=ResourcePoolsConfig)
    # Scheduling-fairness: a READY task skipped this many scheduling rounds
    # is dispatched next regardless of prefix affinity. 0 means "always
    # prefer the oldest waiter", which is valid.
    starvation_rounds: int = Field(default=8, ge=0)

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

    # These two describe invariants the parallel design depends on, so the
    # only supported values are the ones it enforces. Accepting anything
    # else and ignoring it would promise isolation the harness does not
    # deliver — better to fail at load with an explanation.
    @field_validator("task_worktrees")
    @classmethod
    def _worktrees_required(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "git.task_worktrees cannot be disabled: parallel tasks would "
                "share one working tree and overwrite each other. Set "
                "parallelism.max_parallel_tasks: 1 if you want sequential runs."
            )
        return value

    @field_validator("integration_strategy")
    @classmethod
    def _serialized_only(cls, value: str) -> str:
        if value != "serialized":
            raise ValueError(
                f"unsupported git.integration_strategy '{value}'; only "
                "'serialized' is implemented (concurrent merges into the "
                "integration branch are never safe)."
            )
        return value


class DatabaseConfig(BaseModel):
    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5000
    # All writes go through one serialized writer (process-wide lock).
    # Parallel agent coroutines/threads never race write transactions.
    single_writer: bool = True

    @field_validator("single_writer")
    @classmethod
    def _single_writer_required(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "database.single_writer cannot be disabled: parallel agent "
                "coroutines and worker threads would interleave write "
                "transactions on one SQLite connection."
            )
        return value


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
