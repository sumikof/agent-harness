"""Diagnostician: analyzes repeated failures on a task.

Read-only. Recommends RETRY / SPLIT / REPLAN / BLOCKED.
"""

from ..artifacts.schemas import Diagnosis
from ..orchestrator.state_machine import Role
from .base import RoleSpec

SPEC = RoleSpec(
    role=Role.DIAGNOSTICIAN,
    prompt_file="diagnostician.md",
    output_model=Diagnosis,
    output_artifact="diagnosis.json",
    mutates_repo=False,
)
