"""Prefix-cache prompt-stability guarantees (spec items 13-16, 41)."""

from pathlib import Path

from harness.context.attempt_context import AttemptContext
from harness.context.builder import ContextBuilder
from harness.context.prefix import canonical_json, prefix_group_key
from harness.context.project_context import ProjectContext
from harness.context.task_context import TaskContext
from harness.orchestrator.state_machine import Role

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "harness" / "prompts"


def make_project(**overrides) -> ProjectContext:
    values = dict(
        name="proj", goal="do things", repository_path="/anywhere",
        base_branch="main", forbidden_operations=["no git"],
    )
    values.update(overrides)
    return ProjectContext(**values)


def test_canonical_json_is_deterministic():
    a = canonical_json({"b": 1, "a": [2, 1], "nested": {"y": 1, "x": 2}})
    b = canonical_json({"nested": {"x": 2, "y": 1}, "a": [2, 1], "b": 1})
    assert a == b
    # fixed separators, sorted keys
    assert a == '{"a":[2,1],"b":1,"nested":{"x":2,"y":1}}'


def test_prefix_group_key_stable_for_identical_context():
    kwargs = dict(model="m", agent_profile_hash="p", project_context_hash="c",
                  tool_schema_hash="t", common_prompt_hash="h")
    assert prefix_group_key(**kwargs) == prefix_group_key(**kwargs)
    changed = dict(kwargs, tool_schema_hash="other")
    assert prefix_group_key(**kwargs) != prefix_group_key(**changed)


def test_builder_key_stable_and_role_separated():
    builder = ContextBuilder(PROMPTS)
    project = make_project()
    key1 = builder.compute_prefix_group_key(
        model="qwen", agent_profile_hash="p1", project=project,
        role=Role.DEVELOPER, prompt_file="developer.md", tool_schema_hash="t")
    key2 = builder.compute_prefix_group_key(
        model="qwen", agent_profile_hash="p1", project=project,
        role=Role.DEVELOPER, prompt_file="developer.md", tool_schema_hash="t")
    assert key1 == key2
    reviewer_key = builder.compute_prefix_group_key(
        model="qwen", agent_profile_hash="p2", project=project,
        role=Role.REVIEWER, prompt_file="reviewer.md", tool_schema_hash="t2")
    assert key1 != reviewer_key


def test_volatile_metadata_does_not_change_stable_prefix():
    builder = ContextBuilder(PROMPTS)
    project = make_project()
    task = TaskContext(task_key="T001", title="t", goal="g")

    def stable_of(volatile):
        sections = builder.build_sections(Role.DEVELOPER, project, task,
                                          volatile=volatile)
        return [(n, t) for n, t in sections if n in ContextBuilder.STABLE_SECTIONS]

    a = stable_of({"working_directory": "/w1", "attempt_no": 1, "run_id": 17})
    b = stable_of({"working_directory": "/w2", "attempt_no": 9, "run_id": 4242})
    assert a == b

    key_a = builder.compute_prefix_group_key(
        model="qwen", agent_profile_hash="p", project=project,
        role=Role.DEVELOPER, prompt_file="developer.md")
    key_b = builder.compute_prefix_group_key(
        model="qwen", agent_profile_hash="p", project=project,
        role=Role.DEVELOPER, prompt_file="developer.md")
    assert key_a == key_b


def test_sections_ordered_stable_before_dynamic_with_volatile_at_tail():
    builder = ContextBuilder(PROMPTS)
    sections = builder.build_sections(
        Role.DEVELOPER, make_project(),
        TaskContext(task_key="T001", title="t", goal="g"),
        AttemptContext(attempt_no=2, previous_attempt_summary="failed"),
        extra="extra info",
        volatile={"working_directory": "/w", "attempt_no": 2},
    )
    names = [name for name, _ in sections]
    assert names == [
        "harness_rules", "project_context", "task_context", "attempt_context",
        "extra_context", "volatile_metadata", "assignment",
    ]
    # nothing volatile leaks into the stable head
    stable_text = "\n".join(text for name, text in sections
                            if name in ContextBuilder.STABLE_SECTIONS)
    assert "/w" not in stable_text
    assert "attempt" not in stable_text.lower()


def test_stable_prefix_contains_no_repository_path():
    """Per-worktree paths differ per task; the stable project context must
    not embed one."""
    text = make_project(repository_path="/very/specific/worktree").render()
    assert "/very/specific/worktree" not in text


def test_context_budget_truncates_dynamic_never_stable():
    builder = ContextBuilder(PROMPTS, input_budget_tokens=2000)  # 8000 chars
    project = make_project()
    big = "x" * 60000
    sections = builder.build_sections(
        Role.DEVELOPER, project,
        TaskContext(task_key="T001", title="t", goal="g"),
        extra=big,
    )
    as_dict = dict(sections)
    assert len(as_dict["extra_context"]) < len(big)
    assert "truncated by context budget" in as_dict["extra_context"]
    # stable sections stay byte-identical
    assert as_dict["project_context"] == project.render()
    total = sum(len(t) for t in as_dict.values())
    assert total <= 2000 * 4 + 3000  # budget + slack for other sections
