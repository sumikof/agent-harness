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

import contextlib
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Grace period between SIGTERM and SIGKILL for a timed-out process group.
TERM_GRACE_SECONDS = 5
# How long to wait for the group to disappear after SIGKILL.
KILL_GRACE_SECONDS = 2
POLL_INTERVAL_SECONDS = 0.05


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
    """SIGTERM the whole group, then SIGKILL whatever is still in it.

    Escalation is decided by whether the GROUP still has members, never by
    whether its leader exited: a shell that forwards SIGTERM and quits
    while a descendant traps and ignores it would otherwise leave that
    descendant alive — still writing into the worktree — because the
    leader's exit looked like success.
    """
    pgid = _group_id(process)
    _signal_group(process, pgid, signal.SIGTERM)
    if _wait_for_group_exit(process, pgid, TERM_GRACE_SECONDS):
        return
    logger.warning("process group %s survived SIGTERM; escalating to SIGKILL", pgid)
    _signal_group(process, pgid, signal.SIGKILL)
    if not _wait_for_group_exit(process, pgid, KILL_GRACE_SECONDS):
        logger.error("process group %s still present after SIGKILL", pgid)


def _group_id(process: subprocess.Popen) -> int | None:
    if not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        # The leader is already gone; without its pgid the group cannot be
        # addressed, so fall back to the pid the group was created with
        # (start_new_session makes pid == pgid).
        return process.pid


def _signal_group(process: subprocess.Popen, pgid: int | None, sig: int) -> None:
    if pgid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(pgid, sig)
            return
        except ProcessLookupError:
            return                      # nothing left in the group
        except (PermissionError, OSError) as exc:
            logger.debug("could not signal process group %s: %s", pgid, exc)
    # No process groups available (or signalling them failed): best effort
    # on the direct child only.
    with contextlib.suppress(Exception):
        process.send_signal(sig)


def _group_is_alive(pgid: int | None) -> bool:
    """True while ANY process remains in the group (signal 0 probes it)."""
    if pgid is None or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                     # exists, just not ours to signal
    except OSError:
        return False


def _wait_for_group_exit(
    process: subprocess.Popen, pgid: int | None, timeout: float
) -> bool:
    """Wait until the whole group is gone. Returns False on timeout.

    The leader is reaped first: a zombie still counts as a group member, so
    an unreaped leader would make the group look permanently alive.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0)     # reap the leader if it has exited
        if not _group_is_alive(pgid):
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0)
    return not _group_is_alive(pgid)
