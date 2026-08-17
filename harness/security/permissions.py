"""Role -> tool permission mapping.

Role separation is enforced with tool permissions and hooks, not just
prompts. Read-only roles cannot edit; the Test Engineer may only write
under test directories.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import PurePosixPath

from ..orchestrator.state_machine import Role

READ_ONLY_TOOLS = ["Read", "Glob", "Grep", "Bash"]
WRITE_TOOLS = ["Edit", "Write", "MultiEdit", "NotebookEdit"]

ALLOWED_TOOLS: dict[Role, list[str]] = {
    Role.PLANNER: READ_ONLY_TOOLS,
    Role.ANALYST: READ_ONLY_TOOLS,
    Role.DEVELOPER: READ_ONLY_TOOLS + WRITE_TOOLS,
    Role.TESTER: READ_ONLY_TOOLS + WRITE_TOOLS,  # write paths restricted by hook
    Role.REVIEWER: READ_ONLY_TOOLS,
    Role.DIAGNOSTICIAN: READ_ONLY_TOOLS,
}

# Paths (relative to the repository root) the Test Engineer may write to.
TEST_WRITE_GLOBS = [
    "tests/**",
    "test/**",
    "src/test/**",
    "__tests__/**",
    "**/tests/**",
    "**/test/**",
    "**/__tests__/**",
    "**/*_test.py",
    "**/test_*.py",
    "**/*.test.js",
    "**/*.test.ts",
    "**/*.spec.js",
    "**/*.spec.ts",
    "**/*Test.java",
    "**/*Tests.java",
]


def allowed_tools_for(role: Role) -> list[str]:
    return list(ALLOWED_TOOLS[role])


def can_write(role: Role) -> bool:
    return role in (Role.DEVELOPER, Role.TESTER)


def is_test_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path.replace("\\", "/"))
    text = str(path)
    for pattern in TEST_WRITE_GLOBS:
        if fnmatch(text, pattern):
            return True
    return False


def check_write_path(role: Role, relative_path: str) -> tuple[bool, str]:
    """Decide whether `role` may modify `relative_path` (repo-relative)."""
    if role == Role.DEVELOPER:
        return True, ""
    if role == Role.TESTER:
        if is_test_path(relative_path):
            return True, ""
        return False, f"Test Engineer may only modify test files, not {relative_path}"
    return False, f"role {role} is read-only"
