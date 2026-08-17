"""Task Context: information specific to the current task."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskContext:
    task_key: str
    title: str
    goal: str
    acceptance_criteria: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    task_brief: Optional[dict] = None
    implementation: Optional[dict] = None
    test_report: Optional[dict] = None
    relevant_files: list[str] = field(default_factory=list)

    def render(self, include_brief: bool = True) -> str:
        lines = [
            "## Task Context",
            f"Task: {self.task_key} — {self.title}",
            f"Goal: {self.goal}",
        ]
        if self.acceptance_criteria:
            lines.append("\n### Acceptance Criteria")
            lines.extend(f"- {c}" for c in self.acceptance_criteria)
        if self.dependencies:
            lines.append(f"\nDepends on completed tasks: {', '.join(self.dependencies)}")
        if self.relevant_files:
            lines.append("\n### Relevant files")
            lines.extend(f"- {f}" for f in self.relevant_files)
        if include_brief and self.task_brief:
            lines.append("\n### Task Brief (from Task Analyst)")
            lines.append(json.dumps(self.task_brief, indent=2, ensure_ascii=False))
        if self.implementation:
            lines.append("\n### Implementation summary (from Developer)")
            lines.append(json.dumps(self.implementation, indent=2, ensure_ascii=False))
        if self.test_report:
            lines.append("\n### Test report (from Test Engineer)")
            lines.append(json.dumps(self.test_report, indent=2, ensure_ascii=False))
        return "\n".join(lines)
