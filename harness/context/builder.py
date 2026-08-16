"""Context Builder.

Conversation history is never long-term state: the harness assembles a
fresh prompt for every agent session from the three context layers,
selecting only what the role needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..orchestrator.state_machine import Role
from .attempt_context import AttemptContext
from .project_context import ProjectContext
from .task_context import TaskContext


class ContextBuilder:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir

    def system_prompt(self, prompt_file: str) -> str:
        path = self.prompts_dir / prompt_file
        return path.read_text(encoding="utf-8")

    def build_prompt(
        self,
        role: Role,
        project: ProjectContext,
        task: Optional[TaskContext] = None,
        attempt: Optional[AttemptContext] = None,
        extra: str = "",
    ) -> str:
        """Assemble Project + Task + (needed) Attempt context for one session."""
        sections = [project.render()]

        if task is not None:
            # The Analyst produces the brief, so it doesn't receive one;
            # the Reviewer judges the diff on its own terms but needs all artifacts.
            include_brief = role != Role.ANALYST
            sections.append(task.render(include_brief=include_brief))

        if attempt is not None and attempt.has_content():
            sections.append(attempt.render())

        if extra:
            sections.append(extra)

        sections.append(self._task_instruction(role))
        return "\n\n".join(sections)

    def _task_instruction(self, role: Role) -> str:
        instructions = {
            Role.PLANNER: "Analyze the repository and produce the project plan JSON now.",
            Role.ANALYST: "Investigate the repository for this task and produce the task brief JSON now.",
            Role.DEVELOPER: "Implement the task following the brief. When done, produce the implementation JSON.",
            Role.TESTER: "Review test coverage for this change, add missing tests, and produce the test report JSON.",
            Role.REVIEWER: "Review the change against the acceptance criteria and produce the review JSON.",
            Role.DIAGNOSTICIAN: "Analyze why this task keeps failing and produce the diagnosis JSON.",
        }
        return f"## Your assignment\n{instructions[role]}"


def render_plan_for_replan(existing_tasks: list[dict]) -> str:
    """Extra context handed to the Planner when re-planning."""
    return (
        "## Existing plan state\n"
        "The current plan could not proceed. Completed tasks must NOT be re-planned; "
        "revise only the remaining work.\n"
        "```json\n" + json.dumps(existing_tasks, indent=2, ensure_ascii=False) + "\n```"
    )
