"""Shell command execution that leaves nothing running behind.

`subprocess.run(..., shell=True, timeout=...)` kills only the immediate
shell when the deadline passes. Test runners and build tools routinely
fork workers (pytest-xdist, maven surefire, jest workers, gradle
daemons), and those descendants survive — still writing into a task
worktree that the harness is about to archive, reset, or delete.

Commands therefore start in their own process group (session) and the
WHOLE group is terminated on timeout: SIGTERM first so tools can flush,
then SIGKILL for whatever ignored it. The same helper backs both
agent-issued commands and the harness's deterministic verification, so
neither can strand processes in a worktree.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Grace period between SIGTERM and SIGKILL for a timed-out process group.
TERM_GRACE_SECONDS = 5


@dataclass
class CommandResult:
    exit_code: int
    output: str
    timed_out: bool = False


def run_command(
    command: str,
    cwd: str | Path,
    timeout: int,
    *,
    env: dict | None = None,
) -> CommandResult:
    """Run `command` in its own process group; kill the group on timeout.

    Returns the exit code (-1 on timeout) and the combined output captured
    before the deadline.
    """
    popen_kwargs: dict = {}
    if hasattr(os, "setsid"):
        # POSIX: a new session makes the shell a process-group leader, so
        # killpg reaches every descendant it spawned.
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # Windows
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        command,
        shell=True,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        **popen_kwargs,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("command timed out after %ss; killing its process group: %s",
                       timeout, command)
        _terminate_group(process)
        # Drain whatever the group produced before it died, so the timeout
        # report still carries the diagnostic output.
        try:
            stdout, stderr = process.communicate(timeout=TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        output = f"TIMEOUT after {timeout}s\n{stdout or ''}\n{stderr or ''}"
        return CommandResult(exit_code=-1, output=output, timed_out=True)

    output = (stdout or "") + ("\n" + stderr if stderr else "")
    return CommandResult(exit_code=process.returncode, output=output)


def _terminate_group(process: subprocess.Popen) -> None:
    """SIGTERM the whole group, then SIGKILL anything still alive."""
    for sig, wait in ((signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, 2)):
        if process.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), sig)
            else:  # no process groups available — best effort on the child
                process.kill()
        except (ProcessLookupError, PermissionError, OSError) as exc:
            logger.debug("could not signal process group: %s", exc)
            process.kill()
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue
