"""Workspace-wide process lock.

Exactly one harness process may operate on a workspace at a time. The
lock is an OS-level exclusive flock on <workspace>/harness.lock, held
for the process lifetime:

- a second process starting on the same workspace fails loudly instead
  of "recovering" the first process's live RUNNING agent run and
  resetting its working tree;
- a crashed process releases the lock automatically (the kernel drops
  flocks with the file descriptor), so there are no stale locks to
  clean up.

Within one process the lock is reentrant per workspace (the harness may
construct several orchestrators over the same workspace sequentially,
e.g. resume flows and tests).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import fcntl
except ImportError:  # non-POSIX platform
    fcntl = None
try:
    import msvcrt
except ImportError:  # non-Windows platform
    msvcrt = None

# path -> open fd holding the lock, for this process. Process-global by
# design: the lock's scope IS the process.
_HELD: dict[str, int] = {}


class WorkspaceLocked(Exception):
    """Another harness process is already operating on this workspace."""


class UnsupportedPlatform(Exception):
    """No OS file-locking primitive is available — the exclusion guarantee
    cannot be enforced, so the harness refuses to run (fail-closed)."""


def _lock_fd(fd: int, path: Path) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return
    raise UnsupportedPlatform(
        f"this platform provides neither fcntl nor msvcrt file locking; "
        f"cannot enforce single-process exclusion on {path}. Refusing to run."
    )


def _unlock_fd(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


class WorkspaceLock:
    def __init__(self, lock_path: str | Path):
        self.path = Path(lock_path)

    def acquire(self) -> None:
        key = str(self.path.resolve())
        if key in _HELD:
            return  # reentrant within this process
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            _lock_fd(fd, self.path)
        except UnsupportedPlatform:
            os.close(fd)
            raise
        except OSError:
            os.close(fd)
            raise WorkspaceLocked(
                f"another harness process holds {self.path}. Only one process may "
                "run per workspace; stop it (or wait for it to exit) and rerun."
            )
        # Written AFTER the byte-0 lock region is held; only informational.
        os.write(fd, f" {os.getpid()}".encode("ascii"))
        _HELD[key] = fd

    def release(self) -> None:
        key = str(self.path.resolve())
        fd = _HELD.pop(key, None)
        if fd is not None:
            _unlock_fd(fd)
            os.close(fd)
