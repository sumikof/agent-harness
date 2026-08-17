"""Attempt Context: retry-only information (failures, feedback, diagnosis)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

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

    def has_content(self) -> bool:
        return bool(
            self.current_diff
            or self.verification_failure
            or self.review_feedback
            or self.diagnosis
            or self.previous_attempt_summary
        )

    def render(self) -> str:
        lines = ["## Attempt Context", f"Attempt number: {self.attempt_no}"]
        if self.previous_attempt_summary:
            lines.append("\n### Previous attempt summary")
            lines.append(self.previous_attempt_summary)
        if self.verification_failure:
            lines.append("\n### Verification failure (deterministic — exit codes, not opinions)")
            lines.append(json.dumps(self.verification_failure, indent=2, ensure_ascii=False)[:MAX_LOG_CHARS])
        if self.review_feedback:
            lines.append("\n### Review feedback")
            lines.append(json.dumps(self.review_feedback, indent=2, ensure_ascii=False))
        if self.diagnosis:
            lines.append("\n### Diagnosis")
            lines.append(json.dumps(self.diagnosis, indent=2, ensure_ascii=False))
        if self.current_diff:
            diff = self.current_diff
            if len(diff) > MAX_DIFF_CHARS:
                diff = diff[:MAX_DIFF_CHARS] + "\n... (diff truncated)"
            lines.append("\n### Current uncommitted diff")
            lines.append("```diff\n" + diff + "\n```")
        return "\n".join(lines)
