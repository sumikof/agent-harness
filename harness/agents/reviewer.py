"""Reviewer: independent review with a PASS / REPAIR / REPLAN verdict.

Read-only, fresh context — never shares the Developer's session.
"""

from ..artifacts.schemas import Review
from ..orchestrator.state_machine import Role
from .base import RoleSpec

SPEC = RoleSpec(
    role=Role.REVIEWER,
    prompt_file="reviewer.md",
    output_model=Review,
    output_artifact="review.json",
    mutates_repo=False,
)
