"""Configuration loading and validation."""

from __future__ import annotations

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
    type: str = "claude"
    model: str = "claude-sonnet-5"
    roles: dict[str, RoleProviderConfig] = Field(default_factory=dict)

    def for_role(self, role: str) -> tuple[str, str]:
        """Return (provider_type, model) for a role, falling back to defaults."""
        override = self.roles.get(role)
        if override is None:
            return self.type, self.model
        return override.type or self.type, override.model or self.model


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


class LoggingConfig(BaseModel):
    level: str = "INFO"


class HarnessConfig(BaseModel):
    project: ProjectConfig
    workspace_dir: str = "workspace"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
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
        base = self.config_path.parent if self.config_path else Path.cwd()
        return base / "prompts"


def load_config(path: str | Path) -> HarnessConfig:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = HarnessConfig(**raw)
    config.config_path = path
    return config
