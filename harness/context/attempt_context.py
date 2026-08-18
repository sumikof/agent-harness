"""Attempt Context: retry-only information (failures, feedback, diagnosis)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .prefix import canonical_json

MAX_DIFF_CHARS = 20000
MAX_LOG_CHARS = 6000


@dataclass
class AttemptContext:
    attempt_no: int = 1
    current_diff: str = ""
    verification_failure: Optional[dict] = None
    review_feedback: Optional[dict] = None
    diagnosis: Optional[dict] = None
    previous_attempt_summary: str = ""
    # Set when a reviewed-and-passed change could not be merged into the
    # integration branch: the fresh repair attempt re-implements the task on
    # top of the CURRENT integration HEAD, guided by the original diff.
    integration_conflict: Optional[dict] = None

    def has_content(self) -> bool:
        return bool(
            self.current_diff
            or self.verification_failure
            or self.review_feedback
            or self.diagnosis
            or self.previous_attempt_summary
            or self.integration_conflict
        )

    def render(self) -> str:
        lines = ["## Attempt Context", f"Attempt number: {self.attempt_no}"]
        if self.previous_attempt_summary:
            lines.append("\n### Previous attempt summary")
            lines.append(self.previous_attempt_summary)
        if self.verification_failure:
            lines.append("\n### Verification failure (deterministic — exit codes, not opinions)")
            lines.append(canonical_json(self.verification_failure, indent=2)[:MAX_LOG_CHARS])
        if self.review_feedback:
            lines.append("\n### Review feedback")
            lines.append(canonical_json(self.review_feedback, indent=2))
        if self.diagnosis:
            lines.append("\n### Diagnosis")
            lines.append(canonical_json(self.diagnosis, indent=2))
        if self.integration_conflict:
            lines.append("\n### Integration conflict")
            lines.append(
                "A previously reviewed version of this task conflicted while being "
                "merged into the integration branch. Re-implement the task on top of "
                "the CURRENT repository state; the original change is provided for "
                "reference only — do not apply it blindly."
            )
            lines.append(canonical_json(self.integration_conflict, indent=2)[:MAX_LOG_CHARS])
        if self.current_diff:
            diff = self.current_diff
            if len(diff) > MAX_DIFF_CHARS:
                diff = diff[:MAX_DIFF_CHARS] + "\n... (diff truncated)"
            lines.append("\n### Current uncommitted diff")
            lines.append("```diff\n" + diff + "\n```")
        return "\n".join(lines)
