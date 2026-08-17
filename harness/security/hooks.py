"""SDK-agnostic security decisions + Claude Agent SDK PreToolUse hook adapter.

`decide_tool_use` is pure logic (unit-testable, provider-independent).
`build_pretooluse_hook` wraps it in the Claude Agent SDK hook signature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from ..orchestrator.state_machine import Role
from .commands import check_command, find_write_hint
from .permissions import WRITE_TOOLS, check_write_path


class RepeatActionGuard:
    """Detects an agent stuck repeating the identical tool call.

    Consecutive identical (tool name + canonicalized arguments) calls are
    counted: at `warn_after` a warning is recorded, at `abort_after` the
    call is denied and the run is flagged LOOP_DETECTED. The guard only
    observes and denies — the Orchestrator decides the final transition.
    Only wired on providers whose capabilities include pre_tool_hook.
    """

    def __init__(
        self,
        warn_after: int = 3,
        abort_after: int = 5,
        exempt_tools: tuple[str, ...] | list[str] = (),
    ):
        self.warn_after = warn_after
        self.abort_after = abort_after
        self.exempt_tools = set(exempt_tools)
        self._last_hash: Optional[str] = None
        self._count = 0
        self.warnings: list[str] = []
        self.loop_detected = False

    @staticmethod
    def canonical_hash(tool_name: str, tool_input: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"tool": tool_name, "input": tool_input}, sort_keys=True, ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def observe(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        if tool_name in self.exempt_tools:
            self._last_hash = None
            self._count = 0
            return True, ""
        digest = self.canonical_hash(tool_name, tool_input)
        if digest == self._last_hash:
            self._count += 1
        else:
            self._last_hash = digest
            self._count = 1
        if self._count >= self.abort_after:
            self.loop_detected = True
            return False, (
                f"LOOP_DETECTED: identical {tool_name} call repeated "
                f"{self._count} times in a row"
            )
        if self._count == self.warn_after:
            self.warnings.append(
                f"identical {tool_name} call repeated {self._count} times in a row"
            )
        return True, ""


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


def build_pretooluse_hook(
    role: Role, repo_root: str | Path, guard: Optional[RepeatActionGuard] = None
):
    """Build an async PreToolUse hook callable for the Claude Agent SDK."""

    async def pretooluse_hook(
        input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {}) or {}
        allowed, reason = decide_tool_use(role, tool_name, tool_input, repo_root)
        if allowed and guard is not None:
            allowed, reason = guard.observe(tool_name, tool_input)
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
