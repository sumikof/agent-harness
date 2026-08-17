"""SDK-agnostic security decisions + Claude Agent SDK PreToolUse hook adapter.

`decide_tool_use` is pure logic (unit-testable, provider-independent).
`build_pretooluse_hook` wraps it in the Claude Agent SDK hook signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..orchestrator.state_machine import Role
from .commands import check_command, find_write_hint
from .permissions import WRITE_TOOLS, check_write_path


def decide_tool_use(
    role: Role,
    tool_name: str,
    tool_input: dict[str, Any],
    repo_root: str | Path,
) -> tuple[bool, str]:
    """Return (allowed, reason). Deny wins over anything the prompt says."""
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        decision = check_command(command)
        if not decision.allowed:
            return False, f"Forbidden command: {decision.reason}"
        # Only the Developer may run write-capable shell commands. Everyone
        # else (including the Tester, whose writes must go through the
        # path-checked Edit/Write tools) gets read-only Bash.
        if role != Role.DEVELOPER:
            hint = find_write_hint(command)
            if hint:
                return False, (
                    f"{role.value} has read-only Bash ({hint}); "
                    "use the Edit/Write tools if your role permits file changes"
                )
        return True, ""

    if tool_name in WRITE_TOOLS:
        file_path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if not file_path:
            return True, ""
        repo_root = Path(repo_root).resolve()
        target = Path(file_path)
        if not target.is_absolute():
            target = repo_root / target
        try:
            relative = target.resolve().relative_to(repo_root)
        except ValueError:
            return False, f"Writes outside the repository are forbidden: {file_path}"
        allowed, reason = check_write_path(role, str(relative))
        if not allowed:
            return False, reason
        # Nobody edits git internals directly.
        if str(relative).startswith(".git/") or str(relative) == ".git":
            return False, "Modifying .git internals is forbidden"
        return True, ""

    return True, ""


def build_pretooluse_hook(role: Role, repo_root: str | Path):
    """Build an async PreToolUse hook callable for the Claude Agent SDK."""

    async def pretooluse_hook(
        input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {}) or {}
        allowed, reason = decide_tool_use(role, tool_name, tool_input, repo_root)
        if allowed:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return pretooluse_hook
