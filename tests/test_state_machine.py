import pytest

from harness.orchestrator.state_machine import (
    IllegalTransition,
    ProjectState,
    ReviewVerdict,
    Role,
    TaskState,
    assert_project_transition,
    assert_task_transition,
    decide_after_review,
    decide_after_verification,
)


def test_project_happy_path():
    path = [
        ProjectState.CREATED,
        ProjectState.PLANNING,
        ProjectState.READY,
        ProjectState.RUNNING,
        ProjectState.FINAL_VERIFICATION,
        ProjectState.COMPLETED,
    ]
    for current, target in zip(path, path[1:]):
        assert_project_transition(current, target)


def test_project_illegal_transition():
    with pytest.raises(IllegalTransition):
        assert_project_transition(ProjectState.CREATED, ProjectState.COMPLETED)
    with pytest.raises(IllegalTransition):
        assert_project_transition(ProjectState.COMPLETED, ProjectState.RUNNING)


def test_task_happy_path():
    path = [
        TaskState.PENDING,
        TaskState.READY,
        TaskState.ANALYZING,
        TaskState.EXECUTING,
        TaskState.TESTING,
        TaskState.VERIFYING,
        TaskState.REVIEWING,
        TaskState.COMPLETED,
    ]
    for current, target in zip(path, path[1:]):
        assert_task_transition(current, target)


def test_task_repair_cycle():
    assert_task_transition(TaskState.REVIEWING, TaskState.REPAIR_REQUIRED)
    assert_task_transition(TaskState.REPAIR_REQUIRED, TaskState.EXECUTING)
    assert_task_transition(TaskState.VERIFYING, TaskState.REPAIR_REQUIRED)


def test_task_diagnosis_cycle():
    assert_task_transition(TaskState.FAILED, TaskState.DIAGNOSING)
    assert_task_transition(TaskState.DIAGNOSING, TaskState.READY)
    assert_task_transition(TaskState.DIAGNOSING, TaskState.BLOCKED)


def test_task_illegal():
    with pytest.raises(IllegalTransition):
        assert_task_transition(TaskState.PENDING, TaskState.COMPLETED)
    with pytest.raises(IllegalTransition):
        assert_task_transition(TaskState.COMPLETED, TaskState.EXECUTING)


def test_routing_after_verification():
    assert decide_after_verification(True) == Role.REVIEWER
    assert decide_after_verification(False) == Role.DEVELOPER


def test_routing_after_review():
    assert decide_after_review(ReviewVerdict.PASS) is None
    assert decide_after_review(ReviewVerdict.REPAIR) == Role.DEVELOPER
    assert decide_after_review(ReviewVerdict.REPLAN) == Role.PLANNER
