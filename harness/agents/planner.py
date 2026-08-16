"""Project Planner: decomposes the user request into ordered tasks.

Read-only. Runs at project start and when a REPLAN is requested.
"""

from ..artifacts.schemas import ProjectPlan
from ..orchestrator.state_machine import Role
from .base import RoleSpec

SPEC = RoleSpec(
    role=Role.PLANNER,
    prompt_file="planner.md",
    output_model=ProjectPlan,
    output_artifact="project-plan.json",
    mutates_repo=False,
)
