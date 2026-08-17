"""Task Analyst: investigates one task and produces a Task Brief.

Read-only.
"""

from ..artifacts.schemas import TaskBrief
from ..orchestrator.state_machine import Role
from .base import RoleSpec

SPEC = RoleSpec(
    role=Role.ANALYST,
    prompt_file="analyst.md",
    output_model=TaskBrief,
    output_artifact="task-brief.json",
    mutates_repo=False,
)
