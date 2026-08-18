"""PrefixAffinityGate: cap, FIFO, affinity, starvation (spec items 16, 23, 55)."""

import asyncio

import pytest

from harness.orchestrator.resources import PrefixAffinityGate


async def test_cap_is_never_exceeded():
    gate = PrefixAffinityGate(slots=3)
    active = 0
    peak = 0

    async def worker():
        nonlocal active, peak
        await gate.acquire("g")
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        gate.release("g")

    await asyncio.gather(*(worker() for _ in range(20)))
    assert peak <= 3
    assert gate.in_flight == 0


async def test_fifo_when_no_affinity():
    gate = PrefixAffinityGate(slots=1)
    order: list[int] = []
    await gate.acquire(None)

    async def waiter(i):
        await gate.acquire(None)
        order.append(i)
        gate.release(None)

    tasks = []
    for i in range(4):
        tasks.append(asyncio.create_task(waiter(i)))
        await asyncio.sleep(0)  # deterministic enqueue order
    gate.release(None)
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2, 3]


async def test_affinity_preference_over_fifo():
    gate = PrefixAffinityGate(slots=1, starvation_rounds=100)
    order: list[str] = []
    await gate.acquire("A")

    async def waiter(name, group):
        await gate.acquire(group)
        order.append(name)
        gate.release(group)

    tasks = [asyncio.create_task(waiter("b-first", "B"))]
    await asyncio.sleep(0)
    tasks.append(asyncio.create_task(waiter("a-second", "A")))
    await asyncio.sleep(0)
    gate.release("A")  # releasing group A -> the A waiter goes first
    await asyncio.gather(*tasks)
    assert order[0] == "a-second"
    assert gate.affinity_hits >= 1


async def test_starvation_prevention_beats_affinity():
    gate = PrefixAffinityGate(slots=1, starvation_rounds=2)
    order: list[str] = []
    await gate.acquire("A")

    async def waiter(name, group):
        await gate.acquire(group)
        order.append(name)
        await asyncio.sleep(0)
        gate.release(group)

    tasks = [asyncio.create_task(waiter("b-starved", "B"))]
    await asyncio.sleep(0)
    for i in range(4):
        tasks.append(asyncio.create_task(waiter(f"a{i}", "A")))
        await asyncio.sleep(0)
    gate.release("A")
    await asyncio.gather(*tasks)
    # The B waiter was skipped by affinity at most starvation_rounds times.
    assert order.index("b-starved") <= 2


async def test_cancelled_waiter_releases_cleanly():
    gate = PrefixAffinityGate(slots=1)
    await gate.acquire("A")
    waiter = asyncio.create_task(gate.acquire("B"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    gate.release("A")
    # slot is free again
    await gate.acquire("C")
    gate.release("C")
    assert gate.in_flight == 0
