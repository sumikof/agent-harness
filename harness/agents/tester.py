"""Test Engineer: independently reviews and strengthens tests.

May only write files under test directories (enforced by hooks).
"""

from ..artifacts.schemas import TestReport
from ..orchestrator.state_machine import Role
from .base import RoleSpec

SPEC = RoleSpec(
    role=Role.TESTER,
    prompt_file="tester.md",
    output_model=TestReport,
    output_artifact="test-result.json",
    mutates_repo=True,
)
