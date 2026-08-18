"""Deterministic verification.

The verifier is NOT an LLM. An agent saying "tests passed" is never the
official success signal — the harness runs the configured commands and
checks process exit codes itself. Success == every exit code 0.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from ..artifacts.schemas import VerificationResult, VerificationStep
from ..config import VerificationConfig
from ..process import run_command
from . import java, node, python

TAIL_CHARS = 4000


class VerificationRunner:
    def __init__(self, config: VerificationConfig, repo_path: Path, logs_dir: Path):
        self.config = config
        self.repo_path = repo_path
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def commands(self) -> list[str]:
        if self.config.commands:
            return list(self.config.commands)
        language = self.config.language.lower()
        if language == "java":
            return java.default_commands(self.repo_path)
        if language == "python":
            return python.default_commands(self.repo_path)
        if language in ("node", "javascript", "typescript"):
            return node.default_commands(self.repo_path)
        return []

    def run(self, label: str = "verify", on_step=None) -> VerificationResult:
        """Run the configured commands; success == every exit code 0.

        `on_step` (if given) is called after each completed command — the
        orchestrator uses it to durably record the worktree state the
        commands have produced so far, keeping mid-verification crashes
        recoverable at command granularity.
        """
        steps: list[VerificationStep] = []
        passed = True
        for index, command in enumerate(self.commands()):
            step = self._run_command(command, f"{label}-{index}")
            steps.append(step)
            if on_step is not None:
                on_step()
            if step.exit_code != 0:
                passed = False
                break  # fail fast; later steps depend on earlier ones
        return VerificationResult(passed=passed, steps=steps)

    def _run_command(self, command: str, log_name: str) -> VerificationStep:
        log_file = self.logs_dir / f"{log_name}.log"
        started = time.monotonic()
        # Own process group, killed as a group on timeout: build/test workers
        # must not outlive the step and keep writing into the worktree.
        result = run_command(command, self.repo_path, self.config.timeout_seconds)
        exit_code = result.exit_code
        output = result.output
        duration = time.monotonic() - started
        # Full output is always retained on disk; only a bounded tail (plus
        # hash + size for integrity/locating) travels into agent context.
        log_file.write_text(f"$ {command}\nexit: {exit_code}\n\n{output}", encoding="utf-8")
        return VerificationStep(
            command=command,
            exit_code=exit_code,
            duration_seconds=round(duration, 2),
            log_file=str(log_file),
            tail=output[-TAIL_CHARS:],
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            output_bytes=len(output.encode("utf-8")),
        )
