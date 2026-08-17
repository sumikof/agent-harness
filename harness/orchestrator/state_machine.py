"""Deterministic state machines for project / task lifecycle.

The next agent to run is NEVER decided by an LLM. All workflow control
lives in plain Python here and in task_runner.py / project.py.
"""

from __future__ import annotations

from enum import StrEnum


class ProjectState(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    READY = "READY"
    RUNNING = "RUNNING"
    FINAL_VERIFICATION = "FINAL_VERIFICATION"
    COMPLETED = "COMPLETED"
    # Exception states
    BLOCKED = "BLOCKED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    REPLANNING = "REPLANNING"


class TaskState(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    ANALYZING = "ANALYZING"
    EXECUTING = "EXECUTING"
    TESTING = "TESTING"
    VERIFYING = "VERIFYING"
    REVIEWING = "REVIEWING"
    COMPLETED = "COMPLETED"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    FAILED = "FAILED"
    DIAGNOSING = "DIAGNOSING"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class AttemptState(StrEnum):
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class Role(StrEnum):
    PLANNER = "planner"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    DIAGNOSTICIAN = "diagnostician"


class ReviewVerdict(StrEnum):
    PASS = "PASS"
    REPAIR = "REPAIR"
    REPLAN = "REPLAN"


class DiagnosisVerdict(StrEnum):
    RETRY = "RETRY"
    SPLIT = "SPLIT"
    REPLAN = "REPLAN"
    BLOCKED = "BLOCKED"


PROJECT_TRANSITIONS: dict[ProjectState, set[ProjectState]] = {
    ProjectState.CREATED: {ProjectState.PLANNING, ProjectState.FAILED},
    ProjectState.PLANNING: {ProjectState.READY, ProjectState.FAILED, ProjectState.BLOCKED},
    ProjectState.READY: {ProjectState.RUNNING, ProjectState.PAUSED},
    ProjectState.RUNNING: {
        ProjectState.FINAL_VERIFICATION,
        ProjectState.REPLANNING,
        ProjectState.BLOCKED,
        ProjectState.PAUSED,
        ProjectState.FAILED,
    },
    ProjectState.REPLANNING: {ProjectState.RUNNING, ProjectState.READY, ProjectState.BLOCKED, ProjectState.FAILED},
    ProjectState.FINAL_VERIFICATION: {ProjectState.COMPLETED, ProjectState.REPLANNING, ProjectState.FAILED},
    ProjectState.BLOCKED: {ProjectState.RUNNING, ProjectState.REPLANNING, ProjectState.FAILED, ProjectState.PAUSED},
    ProjectState.PAUSED: {ProjectState.RUNNING, ProjectState.READY, ProjectState.FAILED},
    ProjectState.COMPLETED: set(),
    ProjectState.FAILED: set(),
}


TASK_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.PENDING: {TaskState.READY, TaskState.SKIPPED, TaskState.BLOCKED},
    TaskState.READY: {TaskState.ANALYZING, TaskState.SKIPPED, TaskState.BLOCKED},
    TaskState.ANALYZING: {TaskState.EXECUTING, TaskState.FAILED, TaskState.BLOCKED},
    TaskState.EXECUTING: {TaskState.TESTING, TaskState.FAILED, TaskState.BLOCKED},
    TaskState.TESTING: {TaskState.VERIFYING, TaskState.FAILED, TaskState.BLOCKED},
    TaskState.VERIFYING: {TaskState.REVIEWING, TaskState.REPAIR_REQUIRED, TaskState.FAILED},
    TaskState.REVIEWING: {TaskState.COMPLETED, TaskState.REPAIR_REQUIRED, TaskState.FAILED, TaskState.BLOCKED},
    TaskState.REPAIR_REQUIRED: {TaskState.EXECUTING, TaskState.FAILED, TaskState.DIAGNOSING},
    TaskState.FAILED: {TaskState.DIAGNOSING, TaskState.BLOCKED, TaskState.READY},
    TaskState.DIAGNOSING: {TaskState.READY, TaskState.BLOCKED, TaskState.PENDING},
    TaskState.COMPLETED: set(),
    TaskState.BLOCKED: {TaskState.READY, TaskState.SKIPPED},
    TaskState.SKIPPED: set(),
}


class IllegalTransition(Exception):
    pass


def assert_project_transition(current: ProjectState, target: ProjectState) -> None:
    if target not in PROJECT_TRANSITIONS.get(current, set()):
        raise IllegalTransition(f"project: {current} -> {target}")


def assert_task_transition(current: TaskState, target: TaskState) -> None:
    if target not in TASK_TRANSITIONS.get(current, set()):
        raise IllegalTransition(f"task: {current} -> {target}")


def decide_after_verification(verification_ok: bool) -> Role | None:
    """After deterministic verification: fail -> fresh Developer, ok -> Reviewer."""
    return Role.REVIEWER if verification_ok else Role.DEVELOPER


def decide_after_review(verdict: ReviewVerdict) -> Role | None:
    """After review: PASS -> commit (None), REPAIR -> Developer, REPLAN -> Planner."""
    if verdict == ReviewVerdict.PASS:
        return None
    if verdict == ReviewVerdict.REPAIR:
        return Role.DEVELOPER
    return Role.PLANNER
