"""Harness-level invariant validation.

Runs against the persistent state plane (SQLite + event ledger + git +
artifacts) and reports violations. Violations are never silently
repaired here: ERROR-severity findings block the project loudly, and
WARNING-severity findings (typically pre-migration data that predates an
evidence rule) are recorded as events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..database.connection import Database
from ..database.event_repository import EventRepository
from ..git.repository import GitRepository


class Severity(StrEnum):
    ERROR = "ERROR"      # state cannot be trusted; block instead of running
    WARNING = "WARNING"  # suspicious but explainable (e.g. legacy data)


@dataclass
class Violation:
    code: str
    severity: Severity
    message: str


class InvariantChecker:
    def __init__(self, db: Database, git: GitRepository | None = None):
        self.db = db
        self.git = git
        self.events = EventRepository(db)

    def check_all(self, project_id: int | None = None) -> list[Violation]:
        violations: list[Violation] = []
        violations += self.check_event_streams()
        violations += self.check_single_running_agent()
        violations += self.check_running_runs_have_provenance()
        violations += self.check_operation_links()
        violations += self.check_completed_tasks(project_id)
        return violations

    # -- ledger ------------------------------------------------------------

    def check_event_streams(self) -> list[Violation]:
        return [
            Violation(
                "EVENT_SEQ_GAP",
                Severity.ERROR,
                f"event stream {gap['stream_type']}/{gap['stream_id']} has "
                f"{gap['count']} events but seq range {gap['min_seq']}..{gap['max_seq']}",
            )
            for gap in self.events.find_stream_gaps()
        ]

    # -- agent runs --------------------------------------------------------

    def check_single_running_agent(self) -> list[Violation]:
        rows = self.db.query_all("SELECT id FROM agent_runs WHERE status = 'RUNNING'")
        if len(rows) > 1:
            return [
                Violation(
                    "CONCURRENT_AGENT_RUNS",
                    Severity.ERROR,
                    f"{len(rows)} agent runs RUNNING simultaneously "
                    f"(ids {[r['id'] for r in rows]}); the harness runs at most one",
                )
            ]
        return []

    def check_running_runs_have_provenance(self) -> list[Violation]:
        violations = []
        for row in self.db.query_all("SELECT * FROM agent_runs WHERE status = 'RUNNING'"):
            if not row["context_manifest_path"]:
                violations.append(Violation(
                    "RUN_WITHOUT_MANIFEST", Severity.ERROR,
                    f"agent run {row['id']} is RUNNING without a ContextManifest",
                ))
            elif not Path(row["context_manifest_path"]).exists():
                violations.append(Violation(
                    "MANIFEST_ARTIFACT_MISSING", Severity.ERROR,
                    f"agent run {row['id']} references missing manifest "
                    f"{row['context_manifest_path']}",
                ))
            if not row["resolved_spec"]:
                violations.append(Violation(
                    "RUN_WITHOUT_RESOLVED_SPEC", Severity.ERROR,
                    f"agent run {row['id']} is RUNNING without a ResolvedAgentRunSpec",
                ))
            if not row["dispatch_operation_id"]:
                violations.append(Violation(
                    "RUN_WITHOUT_DISPATCH_INTENT", Severity.ERROR,
                    f"agent run {row['id']} is RUNNING without an AGENT_DISPATCH intent",
                ))
        return violations

    # -- operations --------------------------------------------------------

    def check_operation_links(self) -> list[Violation]:
        """Every *_RESULT ledger event must point at a journaled operation."""
        violations = []
        rows = self.db.query_all(
            """
            SELECT e.id, e.event_type, e.operation_id FROM events e
            WHERE e.event_type LIKE '%_RESULT' AND e.operation_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM operations o WHERE o.operation_id = e.operation_id
              )
            """
        )
        for row in rows:
            violations.append(Violation(
                "RESULT_WITHOUT_INTENT", Severity.ERROR,
                f"event {row['id']} ({row['event_type']}) references operation "
                f"{row['operation_id']} that has no journaled intent",
            ))
        return violations

    # -- completed tasks ---------------------------------------------------

    def check_completed_tasks(self, project_id: int | None = None) -> list[Violation]:
        violations = []
        sql = "SELECT * FROM tasks WHERE status = 'COMPLETED'"
        params: tuple = ()
        if project_id is not None:
            sql += " AND project_id = ?"
            params = (project_id,)
        for task in self.db.query_all(sql, params):
            passed = self.db.query_one(
                "SELECT id FROM task_attempts WHERE task_id = ? AND status = 'PASSED' "
                "ORDER BY attempt_no DESC LIMIT 1",
                (task["id"],),
            )
            if passed is None:
                violations.append(Violation(
                    "COMPLETED_WITHOUT_PASSED_ATTEMPT", Severity.ERROR,
                    f"task {task['task_key']} is COMPLETED but has no PASSED attempt",
                ))
                continue
            evaluations = {
                row["verdict"]
                for row in self.db.query_all(
                    "SELECT verdict FROM evaluations WHERE attempt_id = ?", (passed["id"],)
                )
            }
            # Deterministic verification evidence was only recorded from the
            # ledger migration on — its absence on old data is a WARNING.
            if "VERIFY_PASS" not in evaluations:
                violations.append(Violation(
                    "COMPLETED_WITHOUT_VERIFICATION_PASS", Severity.WARNING,
                    f"task {task['task_key']} is COMPLETED without recorded "
                    "deterministic verification PASS evidence",
                ))
            if "PASS" not in evaluations:
                violations.append(Violation(
                    "COMPLETED_WITHOUT_REVIEW_PASS", Severity.WARNING,
                    f"task {task['task_key']} is COMPLETED without a recorded "
                    "Reviewer PASS evaluation",
                ))
            commit = task["current_commit"]
            if commit and self.git is not None and not self.git.commit_exists(commit):
                violations.append(Violation(
                    "COMPLETED_COMMIT_UNRESOLVABLE", Severity.ERROR,
                    f"task {task['task_key']} records commit {commit} that does not "
                    "exist in the repository",
                ))
        return violations

    # -- artifacts ---------------------------------------------------------

    def check_artifact_provenance(self, artifact_path: Path) -> list[Violation]:
        """Validate one enveloped artifact against DB + git state."""
        violations = []
        if not artifact_path.exists():
            return [Violation("ARTIFACT_MISSING", Severity.ERROR,
                              f"artifact {artifact_path} does not exist")]
        try:
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [Violation("ARTIFACT_UNREADABLE", Severity.ERROR,
                              f"artifact {artifact_path}: {exc}")]
        if not isinstance(data, dict) or "artifact_type" not in data:
            return []  # legacy artifact without envelope — nothing to validate
        producer = data.get("producer") or {}
        run_id = producer.get("agent_run_id")
        if run_id is not None:
            row = self.db.query_one("SELECT id FROM agent_runs WHERE id = ?", (run_id,))
            if row is None:
                violations.append(Violation(
                    "ARTIFACT_PRODUCER_MISSING", Severity.ERROR,
                    f"artifact {artifact_path} names producer run {run_id}, "
                    "which does not exist",
                ))
        base_commit = data.get("base_commit")
        if base_commit and self.git is not None and not self.git.commit_exists(base_commit):
            violations.append(Violation(
                "ARTIFACT_BASE_COMMIT_UNRESOLVABLE", Severity.ERROR,
                f"artifact {artifact_path} references base commit {base_commit} "
                "not present in the repository",
            ))
        return violations
