"""Runs one agent role as a fresh session and returns its validated artifact.

Owns the mechanics every role invocation shares: budget checks, capability
validation, ContextManifest + ResolvedAgentRunSpec persistence (both are
durable BEFORE dispatch), AGENT_DISPATCH intent/result journaling, DB run
records, prompt assembly, structured-output validation (with one guided
retry), artifact persistence, and provider-failure retries.

Retry layering (see also task_runner.py):
- Provider retry: transient infrastructure failures, bounded, exponential
  backoff, same task attempt. Permanent failures are never provider-retried.
- Episode recovery: crash/interruption handling lives in recovery.py;
  resume is only used when a provider supports it AND the episode is known
  safe to continue — otherwise a fresh session.
- Reasoning retry: verification FAIL / review REPAIR — always a new
  AgentRun with a fresh session and an incremented task attempt
  (decided in task_runner.py, never here).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ..agents.base import (
    AgentRequest,
    AgentResult,
    FailureKind,
    RoleSpec,
    create_runner,
    validate_output,
)
from ..agents.profile import (
    ROLE_REQUIRED_CAPABILITIES,
    AgentProfile,
    RepeatGuardConfig,
    ResolvedAgentRunSpec,
)
from ..artifacts.manager import ArtifactManager, sha256_text
from ..config import HarnessConfig
from ..context.attempt_context import AttemptContext
from ..context.builder import ContextBuilder
from ..context.manifest import ContextManifest, ContextRef
from ..context.project_context import ProjectContext
from ..context.task_context import TaskContext
from ..database.connection import utcnow
from ..database.event_repository import EventRepository, EventType
from ..database.operation_repository import (
    OperationRepository,
    OperationStatus,
    OperationType,
)
from ..database.run_repository import RunRepository
from ..git.repository import GitRepository, WorktreeSnapshot
from ..security.permissions import TEST_WRITE_GLOBS, allowed_tools_for
from .budget import BudgetManager
from .state_machine import Role

logger = logging.getLogger(__name__)

TECHNICAL_RETRIES = 2          # transient provider failures (rate limit, transport, 5xx)
TECHNICAL_RETRY_DELAY = 15.0   # seconds, doubled per retry
SCHEMA_RETRIES = 1             # one guided re-ask if JSON fails validation


class AgentRunFailed(Exception):
    def __init__(self, role: str, detail: str):
        self.role = role
        self.detail = detail
        super().__init__(f"agent run failed ({role}): {detail}")


class AgentConfigurationError(Exception):
    """Provider/profile misconfiguration (e.g. missing capability).

    Never retried at any layer: the run would fail identically every time.
    Surfaces as BLOCKED/FAILED, not as a burned attempt cycle.
    """

    def __init__(self, role: str, detail: str):
        self.role = role
        self.detail = detail
        super().__init__(f"agent configuration error ({role}): {detail}")


class ConcurrentRunError(Exception):
    """Dispatch would violate a concurrency invariant: either the
    configured max_parallel_agent_runs limit, or the hard rule that at
    most ONE mutating agent runs per task attempt (worktree)."""


class AgentInvoker:
    def __init__(
        self,
        config: HarnessConfig,
        context_builder: ContextBuilder,
        artifacts: ArtifactManager,
        runs: RunRepository,
        events: EventRepository,
        budget: BudgetManager,
        operations: Optional[OperationRepository] = None,
        git: Optional[GitRepository] = None,
        llm_gate=None,
        resource_pools=None,
    ):
        self.config = config
        self.context_builder = context_builder
        self.artifacts = artifacts
        self.runs = runs
        self.events = events
        self.budget = budget
        self.operations = operations
        self.git = git
        # The configured LLM pool (ResourcePools.llm). Passed to every local
        # provider so `parallelism.resource_pools.llm` and the scheduler's
        # metrics observe the SAME gate the requests actually go through.
        self.llm_gate = llm_gate or getattr(resource_pools, "llm", None)
        # Host pools travel to the provider so agent-run builds/tests draw
        # from the same limits as harness verification.
        self.resource_pools = resource_pools
        self._admission: Optional[asyncio.Semaphore] = None
        self._admission_loop = None

    # ------------------------------------------------------------------

    def _admission_gate(self) -> asyncio.Semaphore:
        """Waitable admission control for concurrent AgentRuns.

        Reaching `max_parallel_agent_runs` is ordinary contention — the
        dispatch queues here instead of raising. Bound lazily to the running
        loop so one invoker can serve sequential event loops.
        """
        loop = asyncio.get_running_loop()
        if self._admission is None or self._admission_loop is not loop:
            self._admission = asyncio.Semaphore(
                self.config.parallelism.agent_run_limit()
            )
            self._admission_loop = loop
        return self._admission

    def build_profile(self, spec: RoleSpec) -> AgentProfile:
        provider_type, model = self.config.provider.for_role(spec.role.value)
        writable: list[str] = []
        if spec.role == Role.DEVELOPER:
            writable = ["**"]
        elif spec.role == Role.TESTER:
            writable = list(TEST_WRITE_GLOBS)
        return AgentProfile(
            profile_id=f"{spec.role.value}@{provider_type}",
            role=spec.role.value,
            provider=provider_type,
            model=model,
            prompt_file=spec.prompt_file,
            prompt_template_hash=self.context_builder.prompt_template_hash(spec.prompt_file),
            required_capabilities=list(ROLE_REQUIRED_CAPABILITIES[spec.role]),
            allowed_tools=allowed_tools_for(spec.role),
            writable_paths=writable,
            output_schema=spec.output_model.__name__ if spec.output_model else "",
            max_turns=self.config.limits.max_turns,
            timeout_seconds=self.config.limits.agent_run_timeout_seconds,
            budget_usd=self.config.budget.agent_run_usd,
        )

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
        workdir: Optional[Path] = None,
        git: Optional[GitRepository] = None,
        volatile: Optional[dict] = None,
    ) -> BaseModel:
        """Run the role in a fresh session; return its validated output model.

        `workdir`/`git` select the task's isolated worktree; they default
        to the main repository for project-scope roles (Planner).
        """
        self.budget.check_project(project_id)
        if task_id is not None:
            self.budget.check_task(task_id)

        profile = self.build_profile(spec)
        runner = create_runner(profile.provider, self.config.inference,
                               self.llm_gate, self.resource_pools)

        # Capability validation happens BEFORE anything is dispatched. A
        # provider missing a required capability is a configuration error,
        # never a silently degraded run.
        missing = runner.capabilities().missing(profile.required_capabilities)
        if missing:
            raise AgentConfigurationError(
                spec.role.value,
                f"provider '{profile.provider}' lacks required capabilities: {', '.join(missing)}",
            )

        workdir = workdir or self.config.repository_path
        git = git or self.git
        # Volatile metadata travels at the PROMPT TAIL only — the stable
        # prefix (rules + project context) stays byte-identical across
        # sessions of the same role/project for vLLM prefix caching.
        volatile_meta = {"working_directory": str(workdir)}
        if volatile:
            volatile_meta.update(volatile)

        system_prompt = self.context_builder.system_prompt(spec.prompt_file)
        sections = self.context_builder.build_sections(
            spec.role, project_ctx, task_ctx, attempt_ctx, extra,
            volatile=volatile_meta, system_prompt=system_prompt,
        )
        prefix_key = self._prefix_group_key(spec, profile, project_ctx)

        schema_feedback = ""
        last_error = "unknown"
        for schema_round in range(SCHEMA_RETRIES + 1):
            result, run_id = await self._run_with_technical_retries(
                runner,
                profile,
                system_prompt,
                sections,
                schema_feedback,
                spec,
                project_id,
                task_id,
                attempt_id,
                workdir=workdir,
                git=git,
                prefix_group_key=prefix_key,
            )
            if result.status != "COMPLETED":
                raise AgentRunFailed(spec.role.value, result.error or "agent session failed")

            model_obj, validation_error = validate_output(result, spec.output_model)
            if model_obj is not None:
                if artifact_path is not None:
                    attempt_no = None
                    if attempt_ctx is not None:
                        attempt_no = attempt_ctx.attempt_no
                    run_row = self.runs.get(run_id) if run_id is not None else None
                    self.artifacts.save_enveloped(
                        artifact_path,
                        model_obj,
                        project_id=project_id,
                        task_id=task_id,
                        attempt_no=attempt_no,
                        producer_role=spec.role.value,
                        producer_run_id=run_id,
                        base_commit=git.head_commit() if git else None,
                        input_manifest_hash=run_row["context_manifest_hash"]
                        if run_row else None,
                        created_at=utcnow(),
                    )
                return model_obj

            last_error = validation_error or "invalid output"
            logger.warning("%s output invalid (round %d): %s", spec.role, schema_round, last_error)
            schema_feedback = (
                "\n\n## Output correction required\n"
                f"Your previous output could not be used: {last_error}\n"
                "Respond again with ONLY the required JSON object in a ```json code fence."
            )

        raise AgentRunFailed(spec.role.value, f"structured output invalid after retries: {last_error}")

    # ------------------------------------------------------------------

    def _prefix_group_key(self, spec: RoleSpec, profile: AgentProfile,
                          project_ctx: ProjectContext) -> str:
        """PrefixGroupKey for this dispatch: model + role profile + project
        context + tool schema + shared prompt text. Stable for identical
        context; volatile metadata cannot reach it by construction."""
        tool_hash = ""
        if profile.provider in ("openai-compatible", "openai_compatible", "vllm"):
            from ..agents.local_tools import tool_schema_hash

            tool_hash = tool_schema_hash(spec.role)
        return self.context_builder.compute_prefix_group_key(
            model=profile.model,
            agent_profile_hash=profile.profile_hash(),
            project=project_ctx,
            role=spec.role,
            prompt_file=spec.prompt_file,
            tool_schema_hash=tool_hash,
        )

    def _persist_manifest(
        self,
        profile: AgentProfile,
        system_prompt: str,
        sections: list[tuple[str, str]],
        schema_feedback: str,
        prompt: str,
        project_id: int,
        task_id: Optional[int],
        attempt_no: Optional[int],
        git: Optional[GitRepository] = None,
    ) -> tuple[ContextManifest, Path, str]:
        """Write every context section + the manifest to artifact files.

        Raises on any I/O failure — the agent must not start without a
        durable manifest.
        """
        manifest_key = uuid.uuid4().hex
        manifest_dir = self.artifacts.root / "manifests" / manifest_key
        refs: dict[str, ContextRef] = {}
        # The system prompt body is persisted too — a hash alone cannot
        # reconstruct the input once the packaged prompt file changes. The
        # user prompt is the join of the remaining sections in order.
        all_sections = [("system_prompt", system_prompt)] + list(sections)
        if schema_feedback:
            all_sections.append(("correction", schema_feedback))
        for name, text in all_sections:
            section_path = manifest_dir / f"{name}.md"
            self.artifacts.save_text(section_path, text)
            # Stored workspace-relative: a moved/restored workspace resolves
            # them against its current artifacts root (ArtifactManager.resolve).
            refs[name] = ContextRef(
                artifact=self.artifacts.relpath(section_path), sha256=sha256_text(text)
            )
        manifest = ContextManifest(
            project_id=project_id,
            task_id=task_id,
            attempt_no=attempt_no,
            role=profile.role,
            base_commit=git.head_commit() if git else None,
            sections=refs,
            prompt_template_hash=profile.prompt_template_hash,
            agent_profile_hash=profile.profile_hash(),
            prompt_sha256=sha256_text(prompt),
            system_prompt_sha256=sha256_text(system_prompt),
            created_at=utcnow(),
        )
        manifest_path = manifest_dir / "manifest.json"
        self.artifacts.save_model(manifest_path, manifest)
        # Read back: the manifest existing on disk is a dispatch precondition,
        # and the recorded hash covers the actual persisted bytes so a later
        # integrity check compares like with like.
        persisted = manifest_path.read_text(encoding="utf-8")
        if self.artifacts.load_json(manifest_path) is None:
            raise AgentRunFailed(profile.role, "context manifest could not be persisted")
        return manifest, manifest_path, sha256_text(persisted)

    async def _run_with_technical_retries(
        self,
        runner,
        profile: AgentProfile,
        system_prompt: str,
        sections: list[tuple[str, str]],
        schema_feedback: str,
        spec: RoleSpec,
        project_id: int,
        task_id: Optional[int],
        attempt_id: Optional[int],
        workdir: Optional[Path] = None,
        git: Optional[GitRepository] = None,
        prefix_group_key: str = "",
    ) -> tuple[AgentResult, Optional[int]]:
        delay = TECHNICAL_RETRY_DELAY
        prompt = "\n\n".join(text for _, text in sections) + schema_feedback
        workdir = workdir or self.config.repository_path
        git = git or self.git
        attempt_no = None
        if attempt_id is not None:
            row = self.runs.db.query_one(
                "SELECT attempt_no FROM task_attempts WHERE id = ?", (attempt_id,)
            )
            attempt_no = row["attempt_no"] if row else None
        result: AgentResult = AgentResult(status="FAILED", error="not run")
        run_id: Optional[int] = None
        # For mutating roles, freeze the worktree state at session start
        # (e.g. the Tester starts on top of the Developer's uncommitted
        # work) so a transient retry restores THIS state — never bare HEAD,
        # which would erase the previous role's finished changes. Captured
        # from the actual on-disk bytes (tar + deletions), so git clean
        # filters cannot distort what gets restored.
        pre_dispatch_snapshot: Optional[WorktreeSnapshot] = None
        if spec.mutates_repo and git is not None and git.head_commit() is not None:
            try:
                pre_dispatch_snapshot = git.snapshot_worktree_state()
            except Exception as exc:
                logger.warning("could not snapshot worktree before dispatch: %s", exc)
        for attempt in range(TECHNICAL_RETRIES + 1):
            # Every physical provider call spends money — failed runs and
            # schema retries included — so limits are re-checked before each.
            self.budget.check_project(project_id)
            if task_id is not None:
                self.budget.check_task(task_id)

            # 1. ContextManifest — durable before anything else. Failure to
            #    persist it aborts the dispatch entirely.
            manifest, manifest_path, manifest_hash = self._persist_manifest(
                profile, system_prompt, sections, schema_feedback, prompt,
                project_id, task_id, attempt_no, git=git,
            )

            # 2. Resolve the run spec (still nothing dispatched).
            request = AgentRequest(
                role=spec.role,
                system_prompt=system_prompt,
                prompt=prompt,
                cwd=workdir,
                repo_root=workdir,
                model=profile.model,
                max_turns=profile.max_turns,
                timeout_seconds=profile.timeout_seconds,
                profile=profile,
                context_manifest_path=self.artifacts.relpath(manifest_path),
                context_manifest_hash=manifest_hash,
                repeat_guard=RepeatGuardConfig(
                    enabled=self.config.repeat_guard.enabled,
                    warn_after=self.config.repeat_guard.warn_after,
                    abort_after=self.config.repeat_guard.abort_after,
                    exempt_tools=list(self.config.repeat_guard.exempt_tools),
                ),
                prefix_group_key=prefix_group_key,
            )
            resolved: ResolvedAgentRunSpec = await runner.resolve(request)

            # Admission control: a dispatch waits for a free AgentRun slot
            # instead of failing. Normal contention at the configured limit
            # is backpressure, not an error — the DB checks inside stay as
            # invariant guards for state that should now be unreachable.
            async with self._admission_gate():
                # 3. One WRITE-LOCKED transaction: concurrency checks + run row +
                #    resolved spec + dispatch intent. Two invariants are checked:
                #      - RUNNING AgentRun count <= configured max_parallel_agent_runs
                #      - at most ONE RUNNING mutating agent per task attempt
                #        (also DB-enforced by a partial unique index)
                #    Durable COMMIT happens before the side effect (dispatch).
                db = self.runs.db
                with db.transaction(immediate=True):
                    running = self.runs.running_runs()
                    limit = self.config.parallelism.agent_run_limit()
                    if len(running) >= limit:
                        # Unreachable while admission control holds: this is
                        # an invariant guard against state the harness did
                        # not create (e.g. an externally modified database).
                        raise ConcurrentRunError(
                            f"{len(running)} agent runs already RUNNING (limit {limit}) "
                            "despite holding an admission slot; refusing to dispatch"
                        )
                    if spec.mutates_repo and attempt_id is not None:
                        conflicting = self.runs.running_mutating_for_attempt(attempt_id)
                        if conflicting:
                            raise ConcurrentRunError(
                                f"mutating agent run(s) {[r['id'] for r in conflicting]} still "
                                f"RUNNING for attempt {attempt_id}; one worktree, one writer"
                            )
                    try:
                        run_id = self.runs.start_run(
                            project_id,
                            spec.role.value,
                            attempt_id,
                            provider=resolved.provider,
                            model=resolved.model,
                            profile_hash=resolved.profile_hash,
                            profile_version=resolved.profile_version,
                            context_manifest_path=resolved.context_manifest_path,
                            context_manifest_hash=resolved.context_manifest_hash,
                            resolved_spec=resolved.persistable_dump(),
                            mutating=spec.mutates_repo,
                            prefix_group_key=prefix_group_key,
                        )
                    except sqlite3.IntegrityError as exc:
                        # The partial unique index caught a concurrent mutating
                        # RUNNING row for this attempt from another process.
                        raise ConcurrentRunError(
                            f"another mutating agent run became RUNNING concurrently: {exc}"
                        )
                    dispatch_op_id = None
                    if self.operations:
                        base_diff_hash = None
                        if spec.mutates_repo and git is not None:
                            try:
                                base_diff_hash = git.dirty_state_hash()
                            except Exception as exc:
                                logger.warning("could not hash pre-dispatch state: %s", exc)
                        dispatch_op_id = self.operations.record_intent(
                            OperationType.AGENT_DISPATCH,
                            {
                                "role": spec.role.value,
                                "provider": resolved.provider,
                                "model": resolved.model,
                                "technical_attempt": attempt,
                                "run_id": run_id,
                                # Recovery may only reset a diff whose hash it
                                # has durably recorded.
                                "base_diff_sha256": base_diff_hash,
                            },
                            project_id=project_id,
                            task_id=task_id,
                            attempt_id=attempt_id,
                            agent_run_id=run_id,
                        )
                        db.execute(
                            "UPDATE agent_runs SET dispatch_operation_id = ? WHERE id = ?",
                            (dispatch_op_id, run_id),
                        )
                    self.events.emit(
                        EventType.AGENT_STARTED,
                        project_id=project_id,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        agent_run_id=run_id,
                        operation_id=dispatch_op_id,
                        payload={"role": spec.role.value, "run_id": run_id,
                                 "technical_attempt": attempt},
                    )

                # 4. Side effect: the actual agent session.
                result = await runner.run(resolved)

                # A run that tripped the repeat-action guard is never adopted as
                # a success: its output came from a session stuck in a loop.
                if result.loop_detected and result.status == "COMPLETED":
                    result.status = "FAILED"
                    result.error = "LOOP_DETECTED: identical tool call repeated beyond abort threshold"
                # Per-run budget: the SDK offers no mid-run cost cutoff, so the
                # cap is applied to the run's EFFECTIVE status before anything is
                # recorded — run row, operation result and events then agree.
                cap = self.config.budget.agent_run_usd
                cap_exceeded = result.cost_usd > cap
                if cap_exceeded and result.status == "COMPLETED":
                    result.status = "FAILED"
                    result.error = f"run cost ${result.cost_usd:.2f} exceeded agent_run_usd cap ${cap:.2f}"

                # 5. Result — run row, operation result, billing and every event
                #    land in ONE transaction, so a crash right after the provider
                #    returned cannot leave a COMPLETED run with a dangling
                #    PENDING dispatch or unrecorded cost.
                with db.transaction():
                    self.runs.finish_run(
                        run_id,
                        status=result.status,
                        session_id=result.session_id,
                        output_artifact=spec.output_artifact,
                        token_usage=result.token_usage,
                        cost_usd=result.cost_usd,
                        error=result.error,
                    )
                    if self.operations and dispatch_op_id:
                        self.operations.record_result(
                            dispatch_op_id,
                            OperationStatus.COMPLETED
                            if result.status == "COMPLETED"
                            else OperationStatus.FAILED,
                            {"status": result.status, "error": result.error,
                             "cost_usd": result.cost_usd},
                        )
                    self.budget.record_cost(project_id, task_id, result.cost_usd)
                    for warning in result.loop_warnings:
                        self.events.emit(
                            EventType.LOOP_WARNING, project_id=project_id, task_id=task_id,
                            attempt_id=attempt_id, agent_run_id=run_id, payload={"warning": warning},
                        )
                    if result.loop_detected:
                        # The guard only flags; the Orchestrator owns the
                        # transition: the run fails the attempt (reasoning-retry
                        # path — fresh session, possibly diagnosis), never a
                        # provider retry of the same context.
                        self.events.emit(
                            EventType.LOOP_DETECTED, project_id=project_id, task_id=task_id,
                            attempt_id=attempt_id, agent_run_id=run_id,
                            payload={"role": spec.role.value, "run_id": run_id},
                        )
                    if result.telemetry:
                        self.events.emit(
                            EventType.PROVIDER_TELEMETRY, project_id=project_id, task_id=task_id,
                            attempt_id=attempt_id, agent_run_id=run_id, payload=result.telemetry,
                        )
                    self.events.emit(
                        EventType.AGENT_COMPLETED if result.status == "COMPLETED" else EventType.AGENT_FAILED,
                        project_id=project_id,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        agent_run_id=run_id,
                        payload={
                            "role": spec.role.value,
                            "run_id": run_id,
                            "cost_usd": result.cost_usd,
                            "turns": result.num_turns,
                            "error": result.error,
                            "failure_kind": result.failure_kind.value if result.failure_kind else None,
                        },
                    )
                # An over-budget run fails the attempt (normal retry/diagnosis
                # path) instead of being retried — failed runs included, so a
                # high-cost failure is never blindly redispatched. Only the raise
                # lives outside the transaction; the state was recorded above.
                if cap_exceeded:
                    raise AgentRunFailed(
                        spec.role.value,
                        f"run cost ${result.cost_usd:.2f} exceeded agent_run_usd cap ${cap:.2f}",
                    )
                if result.status == "COMPLETED":
                    return result, run_id
                if result.loop_detected:
                    raise AgentRunFailed(
                        spec.role.value,
                        "loop detected (repeated identical tool call) — aborting for a fresh attempt",
                    )
                # Permanent failures (auth, config, missing model) fail
                # identically on retry — surface immediately instead.
                if result.failure_kind == FailureKind.PERMANENT:
                    raise AgentRunFailed(
                        spec.role.value, f"permanent provider failure: {result.error}"
                    )
            if attempt < TECHNICAL_RETRIES:
                # A mutating role may have half-edited the tree before the
                # transient failure; redispatching on top of that would run
                # the same assignment against an unknown base. Archive the
                # partial diff and restore the session-start state first.
                self._restore_worktree_for_retry(spec, run_id, pre_dispatch_snapshot, git)
                logger.warning(
                    "%s failed technically (%s); retrying in %.0fs", spec.role, result.error, delay
                )
                await asyncio.sleep(delay)
                delay *= 2
        return result, run_id

    def _restore_worktree_for_retry(
        self,
        spec: RoleSpec,
        run_id: Optional[int],
        snapshot: Optional[WorktreeSnapshot],
        git: Optional[GitRepository] = None,
    ) -> None:
        """Bring the worktree back to its session-start state (snapshot),
        which may legitimately be dirty — e.g. the Tester runs on top of the
        Developer's uncommitted implementation. Byte-exact both ways: the
        partial work is archived (patch + real-file tar) and the start state
        restored from real-file bytes, immune to git clean filters."""
        git = git or self.git
        if not spec.mutates_repo or git is None:
            return
        try:
            if git.head_commit() is None:
                return
            if snapshot is None:
                raise RuntimeError("no pre-dispatch worktree snapshot available")
            if git.dirty_state_hash() != snapshot.state_hash:
                archive_path = (
                    self.artifacts.root / "diagnostics"
                    / f"{spec.role.value}-run{run_id}-transient-retry.diff"
                )
                diff = git.snapshot_dirty_bytes()
                if diff.strip():
                    self.artifacts.save_bytes(archive_path, diff)
                self.artifacts.archive_worktree_files(
                    archive_path.with_suffix(".files.tar"), git.path,
                    git.changed_paths(),
                )
            git.restore_worktree_state(snapshot)
        except Exception as exc:
            # Without a known base state a blind redispatch is worse than
            # failing the attempt — surface instead of retrying.
            raise AgentRunFailed(
                spec.role.value, f"could not restore worktree before provider retry: {exc}"
            )
