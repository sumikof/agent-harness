#!/usr/bin/env python3
"""Inference benchmark harness for the DGX Spark vLLM deployment.

Measures what the production decision actually depends on — the PRIMARY
score is aggregate useful output tokens/sec at concurrency 16 — never
just single-stream speed, and never Acceptance Rate alone.

Workload: agent-shaped prompts (long stable system/context prefix + a
task-specific tail asking for code), not synthetic random tokens, so
prefix caching and speculative decoding behave as they will in
production.

Modes
    sweep   concurrency 1 / 4 / 8 / 16 (+ 24 / 32 with --saturation)
    prefix  shared-prefix workload (~20K and ~50K token prefixes);
            compare a run with server-side prefix caching ON vs OFF by
            restarting the server between runs and using --label
    soak    sustained 8-16 concurrent agent-like requests for --minutes
    all     sweep + prefix, then write config/inference-tuning.json

Per-point server-side sweeps (MTP OFF/1/2/3/4, max-num-seqs 8/16/24/32,
max-num-batched-tokens 16384/32768/65536, gpu-memory-utilization
0.75-0.90) need a server restart per point: edit deploy/inference/.env,
./start.sh, then run this with --label mtp3 etc. Results accumulate in
benchmark-results.json keyed by label; the tuning profile is written
from the best labeled run at the target concurrency.

vLLM /metrics counters (prefix cache, spec decode) are discovered by
substring — metric names are version-dependent and never hardcoded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import httpx
except ImportError:
    print("pip install httpx", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Agent-shaped workload
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Developer role in a deterministic multi-agent coding harness.
You work inside an isolated git worktree. You never run git lifecycle commands.
You communicate results only through a JSON artifact. Follow the project's
coding conventions, write tests for behavior you add, and keep diffs minimal."""

# A realistic "stable project context" block, repeated to reach the target
# prefix size. Repetition is fine here: the point is prefix length and
# byte-stability, and vLLM caches by token blocks.
CONTEXT_BLOCK = """## Project context
The service is a payment reconciliation pipeline written in Python 3.11.
Modules: ingest/ (bank feed parsers), match/ (transaction matching engine),
ledger/ (double-entry postings), api/ (FastAPI endpoints), tests/.
Conventions: type hints everywhere, dataclasses for value objects, pytest,
no global mutable state, errors are raised not returned. Money is Decimal.
Matching runs in two passes: exact reference match, then fuzzy amount+date.
"""

TASKS = [
    "Add a CSV bank feed parser for the 'Nordbank' format (semicolon-separated, DD.MM.YYYY dates) with tests.",
    "Fix the fuzzy matcher so amounts within 0.5% OR 2 currency units are considered close; update tests.",
    "Add an idempotency key to the POST /reconcile endpoint and return 409 on replays.",
    "Refactor ledger.post_batch to stream postings instead of materializing the full list.",
    "Add structured logging (JSON) to the matching engine with match-pass and candidate counts.",
    "Write property-based tests for Decimal rounding in ledger.allocate_fees.",
    "Add a retry-with-backoff wrapper for the bank feed HTTP client, max 4 attempts.",
    "Implement pagination for GET /transactions (cursor-based, 100 per page).",
]


def build_prompt(task_index: int, prefix_tokens: int) -> str:
    approx_block_tokens = len(CONTEXT_BLOCK) // 4
    repeats = max(1, prefix_tokens // max(1, approx_block_tokens))
    prefix = CONTEXT_BLOCK * repeats
    task = TASKS[task_index % len(TASKS)]
    return (
        f"{prefix}\n## Your task\n{task}\n"
        "Explain your implementation approach step by step, then write the "
        "complete code for the change, then the tests."
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float = 0.0
    total_s: float = 0.0
    output_tokens: int = 0
    input_tokens: int = 0
    error: str = ""

    @property
    def tokens_per_s(self) -> float:
        gen_time = max(self.total_s - self.ttft_s, 1e-6)
        return self.output_tokens / gen_time


@dataclass
class ScenarioResult:
    name: str
    concurrency: int
    wall_s: float
    results: list[RequestResult] = field(default_factory=list)
    metrics_delta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        ok = [r for r in self.results if r.ok]
        errors = [r for r in self.results if not r.ok]
        ttfts = sorted(r.ttft_s for r in ok)
        latencies = sorted(r.total_s for r in ok)
        out_tokens = sum(r.output_tokens for r in ok)

        def pct(values, p):
            if not values:
                return None
            k = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
            return round(values[k], 3)

        itl = []
        for r in ok:
            if r.output_tokens > 1:
                itl.append((r.total_s - r.ttft_s) / (r.output_tokens - 1))
        return {
            "scenario": self.name,
            "concurrency": self.concurrency,
            "requests": len(self.results),
            "errors": len(errors),
            "error_rate": round(len(errors) / max(len(self.results), 1), 4),
            "wall_seconds": round(self.wall_s, 2),
            "aggregate_output_tokens_per_s": round(out_tokens / max(self.wall_s, 1e-6), 1),
            "per_request_output_tokens_per_s_mean": round(
                statistics.mean(r.tokens_per_s for r in ok), 1) if ok else None,
            "ttft_s_p50": pct(ttfts, 50),
            "ttft_s_p95": pct(ttfts, 95),
            "latency_s_p50": pct(latencies, 50),
            "latency_s_p95": pct(latencies, 95),
            "inter_token_latency_ms_mean": round(statistics.mean(itl) * 1000, 2) if itl else None,
            "total_output_tokens": out_tokens,
            "total_input_tokens": sum(r.input_tokens for r in ok),
            "server_metrics_delta": self.metrics_delta,
            "first_errors": [e.error for e in errors[:3]],
        }


class Bench:
    def __init__(self, base_url: str, model: str, max_tokens: int, stream: bool):
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.model = model
        self.max_tokens = max_tokens
        self.stream = stream
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0),
                                        limits=httpx.Limits(max_connections=64))

    async def close(self):
        await self.client.aclose()

    async def one_request(self, prompt: str) -> RequestResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": self.stream,
        }
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        start = time.monotonic()
        try:
            if self.stream:
                return await self._streamed(payload, start)
            response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
            total = time.monotonic() - start
            if response.status_code != 200:
                return RequestResult(ok=False, total_s=total,
                                     error=f"HTTP {response.status_code}")
            usage = response.json().get("usage") or {}
            return RequestResult(
                ok=True, ttft_s=0.0, total_s=total,
                output_tokens=usage.get("completion_tokens", 0) or 0,
                input_tokens=usage.get("prompt_tokens", 0) or 0,
            )
        except Exception as exc:
            return RequestResult(ok=False, total_s=time.monotonic() - start,
                                 error=f"{type(exc).__name__}: {exc}")

    async def _streamed(self, payload: dict, start: float) -> RequestResult:
        ttft = 0.0
        out_tokens = 0
        in_tokens = 0
        async with self.client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload
        ) as response:
            if response.status_code != 200:
                await response.aread()
                return RequestResult(ok=False, error=f"HTTP {response.status_code}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft == 0.0 and chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                        ttft = time.monotonic() - start
                usage = chunk.get("usage")
                if usage:
                    out_tokens = usage.get("completion_tokens", 0) or 0
                    in_tokens = usage.get("prompt_tokens", 0) or 0
        total = time.monotonic() - start
        return RequestResult(ok=True, ttft_s=ttft, total_s=total,
                             output_tokens=out_tokens, input_tokens=in_tokens)

    async def server_metrics(self) -> dict[str, float]:
        """Prefix-cache / spec-decode counters, matched by substring."""
        wanted = ("prefix_cache", "spec_decod", "num_preempt")
        try:
            response = await self.client.get(f"{self.root}/metrics")
            if response.status_code != 200:
                return {}
        except Exception:
            return {}
        out: dict[str, float] = {}
        for line in response.text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{")[0].split(" ")[0]
            if any(w in name for w in wanted):
                try:
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    continue
        return out

    async def run_scenario(
        self, name: str, concurrency: int, total_requests: int,
        prefix_tokens: int, distinct_prefixes: int = 1,
    ) -> ScenarioResult:
        before = await self.server_metrics()
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []

        async def worker(i: int) -> None:
            async with semaphore:
                # distinct_prefixes > 1 simulates several PrefixGroups
                # (roles/projects) sharing the server.
                group = i % distinct_prefixes
                prompt = build_prompt(i, prefix_tokens) + f"\n<!-- group {group} -->"
                results.append(await self.one_request(prompt))

        start = time.monotonic()
        await asyncio.gather(*(worker(i) for i in range(total_requests)))
        wall = time.monotonic() - start
        after = await self.server_metrics()
        delta = {k: round(after.get(k, 0.0) - before.get(k, 0.0), 1)
                 for k in sorted(set(before) | set(after))}
        scenario = ScenarioResult(name=name, concurrency=concurrency,
                                  wall_s=wall, results=results, metrics_delta=delta)
        print(json.dumps(scenario.summary(), indent=2))
        return scenario


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


async def mode_sweep(bench: Bench, args) -> list[dict]:
    concurrencies = [1, 4, 8, 16]
    if args.saturation:
        concurrencies += [24, 32]
    summaries = []
    for c in concurrencies:
        scenario = await bench.run_scenario(
            name=f"sweep-c{c}", concurrency=c,
            total_requests=max(c * args.requests_per_slot, c),
            prefix_tokens=args.prefix_tokens,
        )
        summaries.append(scenario.summary())
    return summaries


async def mode_prefix(bench: Bench, args) -> list[dict]:
    """Shared-prefix workload. Run once with server prefix caching ON and
    once with it OFF (restart the server; use --label cache-on/cache-off)
    to quantify the effect — the harness cannot toggle it client-side."""
    summaries = []
    for prefix_tokens in (20000, 50000):
        scenario = await bench.run_scenario(
            name=f"prefix-{prefix_tokens // 1000}k", concurrency=8,
            total_requests=16, prefix_tokens=prefix_tokens,
        )
        summaries.append(scenario.summary())
    return summaries


async def mode_soak(bench: Bench, args) -> list[dict]:
    """Sustained agent-like load; watches for hangs, errors, and metric
    stagnation. Memory/OOM watching happens server-side (docker stats /
    dmesg) — this reports client-visible health."""
    deadline = time.monotonic() + args.minutes * 60
    concurrency = 12
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    counter = 0

    async def worker() -> None:
        nonlocal counter
        while time.monotonic() < deadline:
            async with semaphore:
                i = counter
                counter += 1
                prompt = build_prompt(i, args.prefix_tokens)
                results.append(await bench.one_request(prompt))

    before = await bench.server_metrics()
    start = time.monotonic()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.monotonic() - start
    after = await bench.server_metrics()
    scenario = ScenarioResult(
        name=f"soak-{args.minutes}m", concurrency=concurrency, wall_s=wall,
        results=results,
        metrics_delta={k: round(after.get(k, 0) - before.get(k, 0), 1)
                       for k in sorted(set(before) | set(after))},
    )
    summary = scenario.summary()
    print(json.dumps(summary, indent=2))
    return [summary]


# ---------------------------------------------------------------------------
# Persistence: results + frozen tuning profile
# ---------------------------------------------------------------------------


def persist(args, summaries: list[dict]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "benchmark-results.json"
    existing = {}
    if results_path.exists():
        existing = json.loads(results_path.read_text())
    runs = existing.get("runs", [])
    runs.append({
        "label": args.label,
        "mode": args.mode,
        "base_url": args.base_url,
        "model": args.model,
        "stream": args.stream,
        # The serving configuration THESE numbers were measured against.
        # Each sweep point restarts the server with a different .env, so a
        # profile written later must not describe someone else's run.
        "server_profile": read_env_profile(),
        "summaries": summaries,
    })
    results_path.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"\nresults appended to {results_path} (label={args.label!r})")

    if args.mode in ("all", "sweep") and args.write_profile:
        # Selected across ALL accumulated runs, not just this invocation.
        write_tuning_profile(out_dir, args, runs)

    warn_regression(runs, summaries)


# The serving settings whose measured values define a tuning profile.
PROFILE_KEYS = (
    "MODEL", "SERVED_MODEL_NAME", "MODEL_REVISION", "VLLM_IMAGE",
    "VLLM_MIN_VERSION", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION", "KV_CACHE_DTYPE",
    "REASONING_PARSER", "TOOL_CALL_PARSER", "SPECULATIVE_CONFIG",
    "TENSOR_PARALLEL_SIZE", "EXTRA_ARGS",
)


def read_env_profile(env_path: Path | None = None) -> dict:
    """The serving configuration as `start.sh` actually sees it.

    The file must be read with SHELL semantics, not split on '=': start.sh
    sources it, so quoting and expansion decide the effective values. A raw
    split would record SPECULATIVE_CONFIG with its surrounding quotes still
    attached — which then fails to parse as JSON in the written profile.
    """
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / "deploy" / "inference" / ".env"
    if not env_path.exists():
        return {}
    # Ask bash for the values it would export, one NUL-separated pair per key.
    program = "set -a; . \"$1\"; set +a; " + "".join(
        f'printf "%s=%s\\0" {key} "${key}"; ' for key in PROFILE_KEYS
    )
    try:
        completed = subprocess.run(
            ["bash", "-c", program, "bash", str(env_path)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"WARNING: could not source {env_path} ({exc}); "
              "serving profile not recorded for this run", file=sys.stderr)
        return {}
    if completed.returncode != 0:
        print(f"WARNING: sourcing {env_path} failed: {completed.stderr.strip()}",
              file=sys.stderr)
        return {}
    env = {}
    for pair in completed.stdout.split("\0"):
        if "=" in pair:
            key, _, value = pair.partition("=")
            if value != "":
                env[key] = value
    return env


TARGET_CONCURRENCY = 16


def score_of(run: dict) -> tuple | None:
    """Primary score of one run: aggregate tok/s at the target concurrency.

    Returns None for runs that cannot be production candidates — no
    measurement at the target concurrency, or errors during it. A point
    that dropped requests is not "known-good" however fast it looked.
    """
    for summary in run.get("summaries", []):
        if summary.get("concurrency") != TARGET_CONCURRENCY:
            continue
        if summary.get("errors"):
            return None
        value = summary.get("aggregate_output_tokens_per_s")
        if value is None:
            return None
        return (value, summary)
    return None


def write_tuning_profile(out_dir: Path, args, runs: list[dict]) -> None:
    """Freeze the BEST measured configuration as the known-good profile.

    The documented workflow restarts the server per sweep point, so the
    most recent invocation is not necessarily the best one — writing it
    unconditionally would promote whatever happened to run last, including
    a slower point or one that dropped requests. The winner is chosen
    across every accumulated labeled run by aggregate throughput at the
    target concurrency, and its OWN recorded serving configuration is what
    gets written. Production pins this file — no startup auto-tuning.
    """
    scored = [(score_of(run), run) for run in runs]
    candidates = []
    incomplete = []
    for score, run in scored:
        if not score:
            continue
        # A run recorded before serving settings were captured (or one whose
        # .env could not be sourced) cannot become the production winner:
        # the profile would be written from hard-coded defaults instead of
        # the configuration that produced the measurement.
        if not run.get("server_profile"):
            incomplete.append(run.get("label"))
            continue
        candidates.append((score[0], score[1], run))
    if incomplete:
        print(
            f"ignoring run(s) {sorted(set(incomplete))} with no recorded serving "
            "configuration; re-measure them to make them eligible",
            file=sys.stderr,
        )
    if not candidates:
        print(
            f"no eligible run measured concurrency {TARGET_CONCURRENCY} without "
            "errors and with a recorded serving configuration; tuning profile "
            "NOT written — rerun the sweep before production",
            file=sys.stderr,
        )
        return
    best_value, best_summary, best_run = max(candidates, key=lambda c: c[0])
    env = best_run.get("server_profile") or {}
    speculative = None
    if env.get("SPECULATIVE_CONFIG"):
        try:
            speculative = json.loads(env["SPECULATIVE_CONFIG"])
        except json.JSONDecodeError:
            speculative = env["SPECULATIVE_CONFIG"]
    profile = {
        "hardware": "DGX Spark",
        "model": env.get("MODEL", "Qwen/Qwen3.6-27B-FP8"),
        "served_model_name": best_run.get("model", args.model),
        "vllm_version": env.get("VLLM_MIN_VERSION", ">=0.19 (record the measured version)"),
        "container": env.get("VLLM_IMAGE", "PIN-THE-VERIFIED-TAG"),
        "benchmark_label": best_run.get("label"),
        "selected_from": sorted({r.get("label") for _, _, r in candidates}),
        "primary_score": {
            "metric": f"aggregate_output_tokens_per_s @ concurrency {TARGET_CONCURRENCY}",
            "value": best_value,
            "error_rate": best_summary.get("error_rate"),
        },
        "profile": {
            "max_model_len": int(env.get("MAX_MODEL_LEN", 65536)),
            "max_num_seqs": int(env.get("MAX_NUM_SEQS", 16)),
            "max_num_batched_tokens": int(env.get("MAX_NUM_BATCHED_TOKENS", 32768)),
            "gpu_memory_utilization": float(env.get("GPU_MEMORY_UTILIZATION", 0.80)),
            "kv_cache_dtype": env.get("KV_CACHE_DTYPE", "auto"),
            "prefix_caching": True,
            "chunked_prefill": True,
            "language_model_only": True,
            "reasoning_parser": env.get("REASONING_PARSER", "qwen3"),
            "tool_call_parser": env.get("TOOL_CALL_PARSER", "qwen3_coder"),
            "speculative": speculative,
        },
    }
    path = out_dir / "inference-tuning.json"
    path.write_text(json.dumps(profile, indent=2))
    print(f"tuning profile written to {path} "
          f"(best: label={best_run.get('label')!r}, {best_value} tok/s)")


def warn_regression(runs: list[dict], current: list[dict]) -> None:
    """Compare against the saved reference baseline (label 'baseline':
    speculative OFF, prefix cache OFF, concurrency 1). Warn — never fail
    CI on absolute tok/s, which is hardware-dependent."""
    baseline_runs = [r for r in runs if r.get("label") == "baseline"]
    if not baseline_runs:
        return
    base = next((s for s in baseline_runs[-1]["summaries"] if s.get("concurrency") == 1), None)
    cur = next((s for s in current if s.get("concurrency") == 1), None)
    if not base or not cur:
        return
    base_tps = base.get("per_request_output_tokens_per_s_mean") or 0
    cur_tps = cur.get("per_request_output_tokens_per_s_mean") or 0
    if base_tps and cur_tps < base_tps:
        print(
            f"WARNING: optimized configuration is SLOWER than the reference "
            f"baseline at concurrency 1 ({cur_tps} < {base_tps} tok/s). "
            f"Check speculative/prefix settings before freezing this profile.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.6-27b-fp8")
    parser.add_argument("--mode", default="sweep",
                        choices=["sweep", "prefix", "soak", "all"])
    parser.add_argument("--label", default="default",
                        help="name of this server configuration point "
                             "(e.g. baseline, mtp2, seqs24, gpu085)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prefix-tokens", type=int, default=8000)
    parser.add_argument("--requests-per-slot", type=int, default=2,
                        help="requests per concurrency slot in sweeps")
    parser.add_argument("--minutes", type=float, default=10.0, help="soak duration")
    parser.add_argument("--saturation", action="store_true", help="add 24/32 sweep points")
    parser.add_argument("--stream", action="store_true", default=True,
                        help="streamed requests (TTFT measurable)")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="non-streaming (production harness mode; no TTFT)")
    parser.add_argument("--output-dir", default="config")
    parser.add_argument("--no-profile", dest="write_profile", action="store_false")
    args = parser.parse_args()

    bench = Bench(args.base_url, args.model, args.max_tokens, args.stream)
    try:
        summaries: list[dict] = []
        if args.mode in ("sweep", "all"):
            summaries += await mode_sweep(bench, args)
        if args.mode in ("prefix", "all"):
            summaries += await mode_prefix(bench, args)
        if args.mode == "soak":
            summaries += await mode_soak(bench, args)
        persist(args, summaries)
    finally:
        await bench.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
