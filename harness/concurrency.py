"""Cancellation-safe bridging between coroutines and worker threads.

Cancelling a coroutine that awaits `asyncio.to_thread` stops the AWAITER,
never the thread. The thread keeps running — still inside `git merge`,
still writing test output into a worktree — while the cancelling side
believes the operation is over and proceeds to release locks or delete
directories underneath it.

Every place the harness runs an uninterruptible side effect off the event
loop therefore goes through `run_thread_uninterruptible`, which waits for
the worker to settle before letting cancellation propagate. The result of
a cancelled operation is deliberately discarded: its journaled intent
stays PENDING and startup recovery reconciles it against real-world state
(git trailers), which is exactly the crash-window path that already
exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


async def run_thread_uninterruptible(
    func: Callable[..., Any], *args: Any, label: str = "operation"
) -> Any:
    """Run `func(*args)` in a worker thread; on cancellation, wait for it.

    The worker is shielded so cancelling the caller cannot orphan it, and
    the caller does not resume unwinding — releasing locks, removing
    worktrees — until the thread has actually finished.
    """
    worker = asyncio.ensure_future(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        logger.warning(
            "%s cancelled; waiting for its worker thread to settle before cleanup",
            label,
        )
        # A second cancellation must not skip the wait either.
        with contextlib.suppress(BaseException):
            await asyncio.wait({worker})
        raise
