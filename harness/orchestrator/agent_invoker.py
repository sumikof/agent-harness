"""Runs one agent role as a fresh session and returns its validated artifact.

Owns the mechanics every role invocation shares: budget checks, DB run
records, prompt assembly, structured-output validation (with one guided
retry), artifact persistence, and technical-failure retries.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ..agents.base import AgentRequest, AgentResult, RoleSpec, create_runner, validate_output
from ..artifacts.manager import ArtifactManager
from ..config import HarnessConfig
from ..context.attempt_context import AttemptContext
from ..context.builder import ContextBuilder
from ..context.project_context import ProjectContext
from ..context.task_context import TaskContext
from ..database.event_repository import EventRepository
from ..database.run_repository import RunRepository
from .budget import BudgetManager

logger = logging.getLogger(__name__)

TECHNICAL_RETRIES = 2          # transient API/transport failures
TECHNICAL_RETRY_DELAY = 15.0   # seconds, doubled per retry
SCHEMA_RETRIES = 1             # one guided re-ask if JSON fails validation


class AgentRunFailed(Exception):
    def __init__(self, role: str, detail: str):
        self.role = role
        self.detail = detail
        super().__init__(f"agent run failed ({role}): {detail}")


class AgentInvoker:
    def __init__(
        self,
        config: HarnessConfig,
        context_builder: ContextBuilder,
        artifacts: ArtifactManager,
        runs: RunRepository,
        events: EventRepository,
        budget: BudgetManager,
    ):
        self.config = config
        self.context_builder = context_builder
        self.artifacts = artifacts
        self.runs = runs
        self.events = events
        self.budget = budget

    async def invoke(
        self,
        spec: RoleSpec,
        project_id: int,
        project_ctx: ProjectContext,
        task_ctx: Optional[TaskContext] = None,
        attempt_ctx: Optional[AttemptContext] = None,
        extra: str = "",
        task_id: Optional[int] = None,
        attempt_id: Optional[int] = None,
        artifact_path: Optional[Path] = None,
    ) -> BaseModel:
        """Run the role in a fresh session; return its validated output model."""
        self.budget.check_project(project_id)
        if task_id is not None:
            self.budget.check_task(task_id)

        provider_type, model = self.config.provider.for_role(spec.role.value)
        runner = create_runner(provider_type)
        system_prompt = self.context_builder.system_prompt(spec.prompt_file)
        prompt = self.context_builder.build_prompt(
            spec.role, project_ctx, task_ctx, attempt_ctx, extra
        )

        schema_feedback = ""
        last_error = "unknown"
        for schema_round in range(SCHEMA_RETRIES + 1):
            result = await self._run_with_technical_retries(
                runner,
                AgentRequest(
                    role=spec.role,
                    system_prompt=system_prompt,
                    prompt=prompt + schema_feedback,
                    cwd=self.config.repository_path,
                    repo_root=self.config.repository_path,
                    model=model,
                    max_turns=self.config.limits.max_turns,
                ),
                spec,
                project_id,
                task_id,
                attempt_id,
            )
            if result.status != "COMPLETED":
                raise AgentRunFailed(spec.role.value, result.error or "agent session failed")

            model_obj, validation_error = validate_output(result, spec.output_model)
            if model_obj is not None:
                if artifact_path is not None:
                    self.artifacts.save_model(artifact_path, model_obj)
                return model_obj

            last_error = validation_error or "invalid output"
            logger.warning("%s output invalid (round %d): %s", spec.role, schema_round, last_error)
            schema_feedback = (
                "\n\n## Output correction required\n"
                f"Your previous output could not be used: {last_error}\n"
                "Respond again with ONLY the required JSON object in a ```json code fence."
            )

        raise AgentRunFailed(spec.role.value, f"structured output invalid after retries: {last_error}")

    async def _run_with_technical_retries(
        self,
        runner,
        request: AgentRequest,
        spec: RoleSpec,
        project_id: int,
        task_id: Optional[int],
        attempt_id: Optional[int],
    ) -> AgentResult:
        delay = TECHNICAL_RETRY_DELAY
        result: AgentResult = AgentResult(status="FAILED", error="not run")
        for attempt in range(TECHNICAL_RETRIES + 1):
            # Every physical provider call spends money — failed runs and
            # schema retries included — so limits are re-checked before each.
            self.budget.check_project(project_id)
            if task_id is not None:
                self.budget.check_task(task_id)
            run_id = self.runs.start_run(project_id, spec.role.value, attempt_id)
            self.events.emit(
                "AGENT_STARTED",
                project_id=project_id,
                task_id=task_id,
                attempt_id=attempt_id,
                payload={"role": spec.role.value, "run_id": run_id, "technical_attempt": attempt},
            )
            result = await runner.run(request)
            self.runs.finish_run(
                run_id,
                status=result.status,
                session_id=result.session_id,
                output_artifact=spec.output_artifact,
                token_usage=result.token_usage,
                cost_usd=result.cost_usd,
                error=result.error,
            )
            self.budget.record_cost(project_id, task_id, result.cost_usd)
            # Per-run budget: the SDK offers no mid-run cost cutoff, so the
            # cap is enforced right after every run — failed ones included.
            # An over-budget run fails the attempt (normal retry/diagnosis
            # path) instead of being retried.
            cap = self.config.budget.agent_run_usd
            if result.cost_usd > cap:
                self.runs.finish_run(
                    run_id, status="FAILED", session_id=result.session_id,
                    token_usage=result.token_usage, cost_usd=result.cost_usd,
                    error=f"run cost exceeded agent_run_usd cap ${cap:.2f}",
                )
                raise AgentRunFailed(
                    spec.role.value,
                    f"run cost ${result.cost_usd:.2f} exceeded agent_run_usd cap ${cap:.2f}",
                )
            self.events.emit(
                "AGENT_COMPLETED" if result.status == "COMPLETED" else "AGENT_FAILED",
                project_id=project_id,
                task_id=task_id,
                attempt_id=attempt_id,
                payload={
                    "role": spec.role.value,
                    "run_id": run_id,
                    "cost_usd": result.cost_usd,
                    "turns": result.num_turns,
                    "error": result.error,
                },
            )
            if result.status == "COMPLETED":
                return result
            if attempt < TECHNICAL_RETRIES:
                logger.warning(
                    "%s failed technically (%s); retrying in %.0fs", spec.role, result.error, delay
                )
                await asyncio.sleep(delay)
                delay *= 2
        return result
