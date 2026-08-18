"""Project Context: long-lived information every agent receives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectContext:
    name: str
    goal: str
    repository_path: str
    base_branch: str
    architecture_rules: list[str] = field(default_factory=list)
    coding_conventions: list[str] = field(default_factory=list)
    forbidden_operations: list[str] = field(default_factory=list)
    important_decisions: list[str] = field(default_factory=list)

    def render(self) -> str:
        # Deliberately NO concrete filesystem path here: under parallel
        # execution each session works in its own worktree, and the path
        # would both be wrong and break the shared cacheable prefix. The
        # actual working directory travels in the volatile metadata tail.
        lines = [
            "## Project Context",
            f"Project: {self.name}",
            f"Goal: {self.goal}",
            "Repository: your current working directory (an isolated checkout of the project)",
            f"Base branch: {self.base_branch}",
        ]
        for title, items in [
            ("Architecture rules", self.architecture_rules),
            ("Coding conventions", self.coding_conventions),
            ("Forbidden operations", self.forbidden_operations),
            ("Important decisions", self.important_decisions),
        ]:
            if items:
                lines.append(f"\n### {title}")
                lines.extend(f"- {item}" for item in items)
        return "\n".join(lines)


DEFAULT_FORBIDDEN_OPERATIONS = [
    "Never run git commit / push / merge / rebase / reset / stash — the harness owns the git lifecycle.",
    "Never modify remotes, credentials, SSH keys, or network configuration.",
    "Never read .env files or secret environment variables.",
    "Never delete files outside the repository working tree.",
]
