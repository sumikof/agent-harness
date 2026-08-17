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

# path -> open fd holding the flock, for this process. Process-global by
# design: the lock's scope IS the process.
_HELD: dict[str, int] = {}


class WorkspaceLocked(Exception):
    """Another harness process is already operating on this workspace."""


class WorkspaceLock:
    def __init__(self, lock_path: str | Path):
        self.path = Path(lock_path)

    def acquire(self) -> None:
        if fcntl is None:
            return  # no flock on this platform; single-process is by convention
        key = str(self.path.resolve())
        if key in _HELD:
            return  # reentrant within this process
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise WorkspaceLocked(
                f"another harness process holds {self.path}. Only one process may "
                "run per workspace; stop it (or wait for it to exit) and rerun."
            )
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode("ascii"))
        _HELD[key] = fd

    def release(self) -> None:
        if fcntl is None:
            return
        key = str(self.path.resolve())
        fd = _HELD.pop(key, None)
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
