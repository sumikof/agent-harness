"""Resource pools separating LLM inference from host-heavy work.

An LLM request consumes GPU batch slots; `mvn test` / `pytest` consume
CPU/RAM. They must be limited independently, so an agent blocked in a
long tool run never holds a GPU inference slot, and a burst of builds
never starves inference:

    resource_pools:
      llm: 16              # in-flight LLM HTTP requests (PrefixAffinityGate)
      heavy_build: 2
      heavy_test: 2
      git_integration: 1   # serialized by design

The LLM pool is a PrefixAffinityGate rather than a plain semaphore: when
several agent sessions are waiting for a slot, one whose PrefixGroupKey
matches the request that just finished is dispatched first — vLLM can
then reuse the cached prompt prefix. Priority order (fixed):

    1. dependency correctness   (handled by the scheduler, not here)
    2. starvation prevention    (a waiter skipped N times goes first)
    3. prefix affinity
    4. FIFO
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import logging
from dataclasses import dataclass

from ..config import ParallelismConfig

logger = logging.getLogger(__name__)


class PrefixAffinityGate:
    """Bounded concurrency gate with prefix-affinity wakeups.

    Deterministic given the acquire/release order: no randomness, no
    timestamps. Fairness is guaranteed by a skip counter — a waiter passed
    over `starvation_rounds` times is scheduled next regardless of
    affinity.
    """

    def __init__(self, slots: int, starvation_rounds: int = 8):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots = slots
        self.starvation_rounds = starvation_rounds
        self._free = slots
        self._waiters: collections.deque = collections.deque()  # [_Waiter]
        self._seq = itertools.count()
        self._last_released_key: str | None = None
        # Telemetry (observability, not control flow).
        self.dispatched_total = 0
        self.affinity_hits = 0

    @dataclass
    class _Waiter:
        future: asyncio.Future
        group_key: str | None
        seq: int
        skips: int = 0

    @property
    def in_flight(self) -> int:
        return self.slots - self._free

    @property
    def queue_depth(self) -> int:
        return len(self._waiters)

    async def acquire(self, group_key: str | None = None) -> None:
        if self._free > 0 and not self._waiters:
            self._free -= 1
            self.dispatched_total += 1
            return
        waiter = self._Waiter(
            future=asyncio.get_running_loop().create_future(),
            group_key=group_key,
            seq=next(self._seq),
        )
        self._waiters.append(waiter)
        try:
            await waiter.future
        except asyncio.CancelledError:
            if waiter.future.done() and not waiter.future.cancelled():
                # Slot was granted concurrently with cancellation — return it.
                self.release(group_key)
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
            raise

    def release(self, group_key: str | None = None) -> None:
        self._last_released_key = group_key
        self._free += 1
        self._wake_next()

    def _wake_next(self) -> None:
        while self._free > 0 and self._waiters:
            chosen = self._choose()
            self._waiters.remove(chosen)
            if chosen.future.done():
                continue
            self._free -= 1
            self.dispatched_total += 1
            if (
                self._last_released_key is not None
                and chosen.group_key == self._last_released_key
            ):
                self.affinity_hits += 1
            chosen.future.set_result(None)

    def _choose(self) -> "_Waiter":
        # 2. starvation prevention: oldest waiter over the skip budget.
        starved = [w for w in self._waiters if w.skips >= self.starvation_rounds]
        if starved:
            return min(starved, key=lambda w: w.seq)
        # 3. prefix affinity: oldest waiter matching the last released key.
        if self._last_released_key is not None:
            matching = [w for w in self._waiters if w.group_key == self._last_released_key]
            if matching:
                chosen = min(matching, key=lambda w: w.seq)
                for w in self._waiters:
                    if w is not chosen and w.seq < chosen.seq:
                        w.skips += 1
                return chosen
        # 4. FIFO.
        return min(self._waiters, key=lambda w: w.seq)


class ResourcePools:
    """Named concurrency limits shared by every task coroutine."""

    def __init__(self, config: ParallelismConfig, llm_max_requests: int | None = None):
        pools = config.resource_pools
        # ONE authoritative in-flight LLM cap. Two settings describe it
        # (`parallelism.resource_pools.llm` and
        # `inference.concurrency.max_requests`); the smaller wins so neither
        # configured ceiling can be exceeded, and a mismatch is reported
        # rather than silently resolved.
        llm_slots = pools.llm
        if llm_max_requests is not None and llm_max_requests != llm_slots:
            llm_slots = min(llm_slots, llm_max_requests)
            logger.warning(
                "parallelism.resource_pools.llm=%d and inference.concurrency.max_requests=%d "
                "disagree; using %d as the single in-flight LLM cap",
                pools.llm, llm_max_requests, llm_slots,
            )
        self.llm = PrefixAffinityGate(llm_slots, config.starvation_rounds)
        self.heavy_build = asyncio.Semaphore(pools.heavy_build)
        self.heavy_test = asyncio.Semaphore(pools.heavy_test)
        # Serialized integration is a hard invariant: max_git_integrations
        # is forced to 1 regardless of configuration, and the
        # IntegrationManager holds its own lock as a second line of defense.
        self.git_integration = asyncio.Semaphore(1)


# Process-wide LLM gate: every runner in this process shares it, so the
# aggregate in-flight request count towards the serving endpoint never
# exceeds the configured cap regardless of how many components dispatch.
_GLOBAL_GATES: dict[tuple[int, int], PrefixAffinityGate] = {}


def global_llm_gate(max_requests: int, starvation_rounds: int = 8) -> PrefixAffinityGate:
    key = (max_requests, starvation_rounds)
    gate = _GLOBAL_GATES.get(key)
    if gate is None:
        gate = PrefixAffinityGate(max_requests, starvation_rounds)
        _GLOBAL_GATES[key] = gate
    return gate
