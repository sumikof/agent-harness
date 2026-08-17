"""Context Builder.

Conversation history is never long-term state: the harness assembles a
fresh prompt for every agent session from the three context layers,
selecting only what the role needs.
"""

from __future__ import annotations

import hashlib
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

    def prompt_template_hash(self, prompt_file: str) -> str:
        return hashlib.sha256(self.system_prompt(prompt_file).encode("utf-8")).hexdigest()

    def build_sections(
        self,
        role: Role,
        project: ProjectContext,
        task: Optional[TaskContext] = None,
        attempt: Optional[AttemptContext] = None,
        extra: str = "",
    ) -> list[tuple[str, str]]:
        """The named context sections one session receives, in prompt order.

        Named so each section can be persisted individually in the
        ContextManifest; build_prompt() joins exactly these texts.
        """
        sections: list[tuple[str, str]] = [("project_context", project.render())]

        if task is not None:
            # The Analyst produces the brief, so it doesn't receive one;
            # the Reviewer judges the diff on its own terms but needs all artifacts.
            include_brief = role != Role.ANALYST
            sections.append(("task_context", task.render(include_brief=include_brief)))

        if attempt is not None and attempt.has_content():
            sections.append(("attempt_context", attempt.render()))

        if extra:
            sections.append(("extra_context", extra))

        sections.append(("assignment", self._task_instruction(role)))
        return sections

    def build_prompt(
        self,
        role: Role,
        project: ProjectContext,
        task: Optional[TaskContext] = None,
        attempt: Optional[AttemptContext] = None,
        extra: str = "",
    ) -> str:
        """Assemble Project + Task + (needed) Attempt context for one session."""
        return "\n\n".join(
            text for _, text in self.build_sections(role, project, task, attempt, extra)
        )

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
