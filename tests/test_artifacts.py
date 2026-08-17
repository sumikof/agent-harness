from harness.artifacts.manager import ArtifactManager
from harness.artifacts.schemas import (
    Diagnosis,
    ProjectPlan,
    Review,
    TaskBrief,
    VerificationResult,
    VerificationStep,
)


def test_save_and_load_model(tmp_path):
    manager = ArtifactManager(tmp_path / "artifacts")
    brief = TaskBrief(task="T001", summary="s", files=["a.py"], verification=["pytest"])
    path = manager.task_artifact_path("T001", "task-brief.json")
    manager.save_model(path, brief)
    loaded = manager.load_model(path, TaskBrief)
    assert loaded.task == "T001"
    assert loaded.files == ["a.py"]


def test_load_missing_returns_none(tmp_path):
    manager = ArtifactManager(tmp_path / "artifacts")
    assert manager.load_json(manager.project_plan_path()) is None


def test_schema_validation():
    review = Review.model_validate(
        {
            "verdict": "REPAIR",
            "blocking_issues": [
                {"file": "a.py", "issue": "bug", "required_fix": "fix it"}
            ],
        }
    )
    assert review.verdict == "REPAIR"

    diagnosis = Diagnosis.model_validate(
        {"recommendation": "SPLIT", "split_tasks": [{"task_key": "T001A", "title": "half"}]}
    )
    assert diagnosis.split_tasks[0].task_key == "T001A"

    plan = ProjectPlan.model_validate(
        {"tasks": [{"task_key": "T001", "title": "t"}]}
    )
    assert plan.tasks[0].dependencies == []

    result = VerificationResult(
        passed=False,
        steps=[VerificationStep(command="pytest", exit_code=1, duration_seconds=1.0)],
    )
    assert not result.passed
