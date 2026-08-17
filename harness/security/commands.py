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
    # Git lifecycle belongs to the harness. `[^|;&]*?` skips global options
    # such as `-C <path>` / `-c <k>=<v>` / `--git-dir=...` that git accepts
    # before the subcommand, so they cannot be used to smuggle a lifecycle
    # command past the hook. False positives (a keyword appearing later in
    # the same segment) are accepted by design.
    (
        r"\bgit\b[^|;&]*?\b(push|commit|merge|rebase|reset|checkout|switch|stash|clean|tag|remote)\b",
        "git lifecycle commands are reserved for the harness",
    ),
    (r"\bgit\b[^|;&]*?\bbranch\b[^|;&]*?(\s-d\b|\s-D\b|--delete)", "branch deletion is forbidden"),
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


# Heuristics that flag a shell command as capable of writing files. Used to
# keep read-only roles (and the Tester, who must edit via the path-checked
# Edit/Write tools) from mutating the repository through Bash. This is
# defense-in-depth, not a sandbox: false positives only force the agent to
# use the dedicated file tools, which are properly permission-checked.
_WRITE_HINTS: list[tuple[re.Pattern, str]] = [
    # Redirections that create/overwrite files. `2>` (fd redirect) and
    # redirects to /dev/null are allowed; `>&`-style fd duplication too.
    (re.compile(r"(?:^|[^&\d])>>?(?!&)\s*(?!/dev/null\b)\S"), "shell redirection writes a file"),
    (re.compile(r"&>\s*(?!/dev/null\b)\S"), "shell redirection writes a file"),
    (re.compile(r"\btee\b"), "tee writes files"),
    (re.compile(r"\bsed\b[^|;&]*\s-i\b"), "sed -i edits files in place"),
    (re.compile(r"\b(perl|python[0-9.]*|ruby)\b[^|;&]*\s-i\b"), "in-place edit flag"),
    (
        re.compile(
            r"\b(mv|cp|rm|touch|mkdir|rmdir|truncate|ln|install|rsync|patch|chmod|chown|dd)\b"
        ),
        "file mutation command",
    ),
]


def find_write_hint(command: str) -> str | None:
    """Return a reason string if the command looks write-capable, else None."""
    normalized = " ".join(command.split())
    for regex, reason in _WRITE_HINTS:
        if regex.search(normalized):
            return reason
    return None
