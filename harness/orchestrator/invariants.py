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
    def __init__(
        self,
        db: Database,
        git: GitRepository | None = None,
        artifacts_root: Path | None = None,
        max_parallel_agent_runs: int = 16,
    ):
        self.db = db
        self.git = git
        self.artifacts_root = Path(artifacts_root) if artifacts_root else None
        self.max_parallel_agent_runs = max_parallel_agent_runs
        self.events = EventRepository(db)

    def _resolve_artifact(self, stored_path: str) -> Path:
        path = Path(stored_path)
        if not path.is_absolute() and self.artifacts_root is not None:
            return self.artifacts_root / path
        return path

    def check_all(self, project_id: int | None = None) -> list[Violation]:
        violations: list[Violation] = []
        violations += self.check_event_streams()
        violations += self.check_agent_run_concurrency()
        violations += self.check_running_runs_have_provenance()
        violations += self.check_operation_links()
        violations += self.check_completed_tasks(project_id)
        violations += self.check_worktrees(project_id)
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

    def check_agent_run_concurrency(self) -> list[Violation]:
        """RUNNING AgentRun count <= configured max_parallel_agent_runs, and
        at most ONE RUNNING mutating agent per task attempt (worktree)."""
        violations: list[Violation] = []
        rows = self.db.query_all("SELECT id FROM agent_runs WHERE status = 'RUNNING'")
        if len(rows) > self.max_parallel_agent_runs:
            violations.append(
                Violation(
                    "CONCURRENT_AGENT_RUNS",
                    Severity.ERROR,
                    f"{len(rows)} agent runs RUNNING simultaneously "
                    f"(ids {[r['id'] for r in rows]}); configured limit is "
                    f"{self.max_parallel_agent_runs}",
                )
            )
        for row in self.db.query_all(
            """
            SELECT attempt_id, COUNT(*) AS n FROM agent_runs
            WHERE status = 'RUNNING' AND mutating = 1 AND attempt_id IS NOT NULL
            GROUP BY attempt_id HAVING COUNT(*) > 1
            """
        ):
            violations.append(
                Violation(
                    "CONCURRENT_MUTATING_RUNS_PER_ATTEMPT",
                    Severity.ERROR,
                    f"attempt {row['attempt_id']} has {row['n']} RUNNING mutating "
                    "agent runs; one worktree admits one writer",
                )
            )
        return violations

    def check_running_runs_have_provenance(self) -> list[Violation]:
        violations = []
        for row in self.db.query_all("SELECT * FROM agent_runs WHERE status = 'RUNNING'"):
            if not row["context_manifest_path"]:
                violations.append(Violation(
                    "RUN_WITHOUT_MANIFEST", Severity.ERROR,
                    f"agent run {row['id']} is RUNNING without a ContextManifest",
                ))
            elif not self._resolve_artifact(row["context_manifest_path"]).exists():
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

    # -- worktrees ---------------------------------------------------------

    def check_worktrees(self, project_id: int | None = None) -> list[Violation]:
        """Parallel-execution worktree invariants:

        - a RUNNING attempt with a mutating agent has a dedicated worktree
        - two RUNNING attempts never share a worktree
        - each recorded worktree exists and has its expected branch checked out
        - each attempt's base_commit resolves in git
        """
        violations: list[Violation] = []
        sql = """
            SELECT a.*, t.task_key, t.project_id FROM task_attempts a
            JOIN tasks t ON t.id = a.task_id
            WHERE a.status = 'RUNNING'
        """
        params: tuple = ()
        if project_id is not None:
            sql += " AND t.project_id = ?"
            params = (project_id,)
        running = self.db.query_all(sql, params)

        seen_worktrees: dict[str, str] = {}
        for attempt in running:
            task_key = attempt["task_key"]
            worktree = attempt["worktree_path"]
            branch = attempt["branch"]
            base_commit = attempt["base_commit"]
            if worktree:
                if worktree in seen_worktrees:
                    violations.append(Violation(
                        "WORKTREE_SHARED", Severity.ERROR,
                        f"RUNNING attempts of {seen_worktrees[worktree]} and {task_key} "
                        f"share worktree {worktree}",
                    ))
                seen_worktrees[worktree] = task_key
                path = Path(worktree)
                if not path.is_dir():
                    violations.append(Violation(
                        "WORKTREE_MISSING", Severity.ERROR,
                        f"RUNNING attempt {attempt['id']} ({task_key}) records worktree "
                        f"{worktree}, which does not exist",
                    ))
                elif branch:
                    checked_out = GitRepository(path).current_branch()
                    if checked_out != branch:
                        violations.append(Violation(
                            "WORKTREE_WRONG_BRANCH", Severity.ERROR,
                            f"worktree {worktree} has '{checked_out}' checked out; "
                            f"attempt {attempt['id']} expects '{branch}'",
                        ))
            else:
                mutating = self.db.query_one(
                    "SELECT 1 FROM agent_runs WHERE attempt_id = ? AND mutating = 1 "
                    "AND status = 'RUNNING'",
                    (attempt["id"],),
                )
                if mutating is not None:
                    violations.append(Violation(
                        "MUTATING_RUN_WITHOUT_WORKTREE", Severity.ERROR,
                        f"RUNNING attempt {attempt['id']} ({task_key}) has a mutating "
                        "agent but no dedicated worktree recorded",
                    ))
            if base_commit and self.git is not None and not self.git.commit_exists(base_commit):
                violations.append(Violation(
                    "ATTEMPT_BASE_COMMIT_UNRESOLVABLE", Severity.ERROR,
                    f"attempt {attempt['id']} ({task_key}) records base commit "
                    f"{base_commit} that does not exist in the repository",
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
