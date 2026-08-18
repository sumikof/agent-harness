"""Parallel Task Scheduler.

Deterministic, dependency-aware, prefix-aware dispatch of READY tasks:

    Task DAG (Planner output, harness-managed)
        │
        ▼
    runnable_tasks()          dependencies satisfied, PENDING/READY
        │
        ▼
    prioritize()              1. dependency correctness (by construction)
        │                     2. starvation prevention (aging)
        ▼                     3. prefix affinity
    up to max_parallel_tasks  4. FIFO (plan sequence)
    concurrent TaskRunner coroutines, each in its own worktree

The DAG is a real scheduling input — a task becomes runnable the moment
its dependencies are COMPLETED/SKIPPED, not when a global sequence
reaches it. The LLM never decides what runs next.

Failure semantics: a task-level failure is handled inside
TaskRunner.run_task (BLOCKED/diagnosis). Only project-fatal conditions
escape a task coroutine (project budget, configuration errors); the
scheduler then STOPS LAUNCHING, drains the already-started tasks
(started operations finish and persist durable results — see
Cancellation in the design notes), and re-raises the first fatal error.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from enum import StrEnum

from ..config import HarnessConfig
from ..context.project_context import ProjectContext
from ..database.event_repository import EventRepository, EventType
from ..database.task_repository import TaskRepository
from .budget import BudgetExceeded, BudgetManager
from .task_runner import TaskOutcome, TaskRunner

logger = logging.getLogger(__name__)


class SchedulerOutcome(StrEnum):
    DONE = "DONE"        # no runnable and no in-flight work remains
    REPLAN = "REPLAN"    # a task requested replanning (drained first)


class ParallelTaskScheduler:
    def __init__(
        self,
        config: HarnessConfig,
        tasks: TaskRepository,
        task_runner: TaskRunner,
        events: EventRepository,
        budget: BudgetManager,
    ):
        self.config = config
        self.tasks = tasks
        self.task_runner = task_runner
        self.events = events
        self.budget = budget
        self.max_parallel = config.parallelism.max_parallel_tasks
        self.starvation_rounds = config.parallelism.starvation_rounds
        self._wait_rounds: dict[int, int] = {}

    async def run(self, project_id: int, project_ctx: ProjectContext) -> SchedulerOutcome:
        """Schedule until the DAG is exhausted or a replan is requested."""
        inflight: dict[asyncio.Task, sqlite3.Row] = {}
        replan_requested = False
        fatal: BaseException | None = None

        while True:
            if fatal is None and not replan_requested:
                try:
                    self.budget.check_project(project_id)
                except BudgetExceeded as exc:
                    fatal = exc
            if fatal is None and not replan_requested:
                running_ids = {row["id"] for row in inflight.values()}
                runnable = [
                    row for row in self.tasks.runnable_tasks(project_id)
                    if row["id"] not in running_ids
                ]
                for row in self._prioritize(runnable):
                    if len(inflight) >= self.max_parallel:
                        break
                    self._wait_rounds.pop(row["id"], None)
                    coro = self.task_runner.run_task(project_id, row, project_ctx)
                    task = asyncio.create_task(coro, name=f"task-{row['task_key']}")
                    inflight[task] = row
                # Aging for fairness: every runnable task NOT dispatched this
                # round accumulates a wait round.
                dispatched = {row["id"] for row in inflight.values()}
                for row in runnable:
                    if row["id"] not in dispatched:
                        self._wait_rounds[row["id"]] = self._wait_rounds.get(row["id"], 0) + 1

            if not inflight:
                break

            self._emit_metrics(project_id, inflight)
            done, _ = await asyncio.wait(inflight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                row = inflight.pop(finished)
                try:
                    outcome = finished.result()
                except BudgetExceeded as exc:
                    # Drain: stop launching, let the started tasks finish
                    # (each persists durable results), then surface.
                    logger.warning("task %s hit project budget: %s", row["task_key"], exc)
                    fatal = fatal or exc
                    continue
                except asyncio.CancelledError:
                    continue
                except BaseException as exc:
                    logger.error("task %s coroutine failed: %s", row["task_key"], exc)
                    fatal = fatal or exc
                    continue
                logger.info("task %s outcome: %s", row["task_key"], outcome)
                if outcome == TaskOutcome.REPLAN:
                    # Replanning rewrites unfinished task definitions — it
                    # must not race in-flight work. Drain first.
                    replan_requested = True
                # COMPLETED / SPLIT / BLOCKED: nothing to do; the next round
                # recomputes the runnable frontier (SPLIT added new tasks).

        if fatal is not None:
            raise fatal
        return SchedulerOutcome.REPLAN if replan_requested else SchedulerOutcome.DONE

    # ------------------------------------------------------------------

    def _emit_metrics(self, project_id: int, inflight: dict) -> None:
        """One observability snapshot per scheduling round (ledger event):
        task/LLM concurrency, queue depths, prefix-affinity effectiveness."""
        gate = self.task_runner.pools.llm
        try:
            self.events.emit(
                EventType.METRICS_SNAPSHOT,
                project_id=project_id,
                payload={
                    "tasks_in_flight": len(inflight),
                    "task_keys": sorted(row["task_key"] for row in inflight.values()),
                    "ready_queue_length": len(self._wait_rounds),
                    "llm_in_flight": gate.in_flight,
                    "llm_queue_depth": gate.queue_depth,
                    "llm_dispatched_total": gate.dispatched_total,
                    "llm_affinity_hits": gate.affinity_hits,
                },
            )
        except Exception as exc:  # observability must never break scheduling
            logger.debug("metrics snapshot failed: %s", exc)

    def _prioritize(self, runnable: list[sqlite3.Row]) -> list[sqlite3.Row]:
        """Dispatch order among dependency-satisfied tasks.

        1. dependency correctness — already guaranteed by runnable_tasks()
        2. starvation prevention — tasks skipped >= starvation_rounds first
        3. prefix affinity — the fine-grained affinity lives in the LLM
           request gate (PrefixAffinityGate); at task granularity all tasks
           of one project share their stable prefix, so plan order is the
           natural grouping
        4. FIFO — plan sequence
        """
        starved = [
            row for row in runnable
            if self._wait_rounds.get(row["id"], 0) >= self.starvation_rounds
        ]
        starved_ids = {row["id"] for row in starved}
        rest = [row for row in runnable if row["id"] not in starved_ids]
        key = lambda row: (row["sequence"], row["id"])  # noqa: E731
        return sorted(starved, key=key) + sorted(rest, key=key)
