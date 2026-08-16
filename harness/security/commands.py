"""Shell command policy.

The harness owns the git lifecycle and the host environment; agents get a
restricted Bash. Deny rules are checked against every command an agent
tries to run, before execution (PreToolUse hook).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns are matched against the whole command string (case-insensitive,
# after whitespace normalization). Kept deliberately broad: false positives
# only cost an agent a denied tool call; false negatives cost repo state.
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    # Git lifecycle belongs to the harness
    (r"\bgit\s+push\b", "git push is reserved for the harness"),
    (r"\bgit\s+commit\b", "git commit is reserved for the harness"),
    (r"\bgit\s+merge\b", "git merge is reserved for the harness"),
    (r"\bgit\s+rebase\b", "git rebase is reserved for the harness"),
    (r"\bgit\s+reset\b", "git reset is reserved for the harness"),
    (r"\bgit\s+checkout\s+", "git checkout is reserved for the harness"),
    (r"\bgit\s+switch\b", "git switch is reserved for the harness"),
    (r"\bgit\s+branch\s+(-d|-D|--delete)\b", "branch deletion is forbidden"),
    (r"\bgit\s+remote\b", "modifying remotes is forbidden"),
    (r"\bgit\s+stash\b", "git stash is reserved for the harness"),
    (r"\bgit\s+clean\b", "git clean is reserved for the harness"),
    (r"\bgit\s+tag\b", "git tag is reserved for the harness"),
    # Destructive / privileged operations
    (r"\bsudo\b", "sudo is forbidden"),
    (r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+(/|~|\$HOME)(\s|$)", "recursive delete of root/home is forbidden"),
    (r"\brm\s+-rf\s+/(\s|$)", "rm -rf / is forbidden"),
    (r"\bmkfs\b", "formatting filesystems is forbidden"),
    (r"\bdd\s+.*of=/dev/", "writing to block devices is forbidden"),
    (r"\bdocker\s+system\s+prune\b", "docker system prune is forbidden"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\bshutdown\b|\breboot\b", "host lifecycle operations are forbidden"),
    # Credentials / secrets
    (r"\bgit\s+config\s+.*credential", "credential changes are forbidden"),
    (r"~/\.ssh|/\.ssh/", "SSH key access is forbidden"),
    (r"\bssh-keygen\b|\bssh-add\b", "SSH key operations are forbidden"),
    (r"(^|[\s/])\.env(\.|\s|$)", "reading .env secrets is forbidden"),
    (r"\bAWS_SECRET|\bGITHUB_TOKEN|\bANTHROPIC_API_KEY", "reading credential variables is forbidden"),
    # Network / system configuration
    (r"\biptables\b|\bufw\b|\bfirewall-cmd\b", "changing network configuration is forbidden"),
    (r"\bifconfig\b.*\s(up|down)\b|\bip\s+link\s+set\b", "changing network interfaces is forbidden"),
    (r"\bcrontab\b", "editing crontab is forbidden"),
    (r"\bchown\s+.*root\b", "chown to root is forbidden"),
]

_COMPILED = [(re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in FORBIDDEN_PATTERNS]


@dataclass
class CommandDecision:
    allowed: bool
    reason: str = ""


def check_command(command: str) -> CommandDecision:
    normalized = " ".join(command.split())
    for regex, reason in _COMPILED:
        if regex.search(normalized):
            return CommandDecision(False, reason)
    return CommandDecision(True)
