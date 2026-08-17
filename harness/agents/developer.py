"""Developer: implements the Task Brief.

May read and write source code. Git lifecycle operations are blocked by
security hooks — the harness owns commits.
"""

from ..artifacts.schemas import Implementation
from ..orchestrator.state_machine import Role
from .base import RoleSpec

SPEC = RoleSpec(
    role=Role.DEVELOPER,
    prompt_file="developer.md",
    output_model=Implementation,
    output_artifact="implementation.json",
    mutates_repo=True,
)
