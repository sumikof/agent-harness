"""Inference endpoint health checks and capability verification.

Capabilities are never assumed: before the harness starts a project on
the local OpenAI-compatible provider, the endpoint must pass

    1. liveness            GET /health (vLLM) or GET /v1/models
    2. model availability  the configured alias is served
    3. simple completion   a trivial prompt returns text + usage
    4. tool calling        the model emits a correct read_file tool call
                           (qwen3_coder parser on the server side)
    5. structured output   the model returns parseable JSON on request
    6. prefix caching      two requests sharing a long prefix; the served
                           /metrics (when reachable) must show prefix
                           cache activity — merely passing the CLI flag
                           is NOT accepted as verification

Failures 1-5 block startup (a misconfigured provider must not burn
long-running tasks). 6 degrades to a loud warning when /metrics is not
reachable from the harness host.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..config import InferenceConfig

logger = logging.getLogger(__name__)

_TOOL_SMOKE_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]

# Long shared prefix for the cache smoke test (needs to exceed the
# server's prefix-match block size comfortably).
_PREFIX_FILLER = ("The harness verifies serving features before use. " * 400).strip()


@dataclass
class HealthReport:
    healthy: bool = False
    model_available: bool = False
    completion_ok: bool = False
    usage_reported: bool = False
    tool_calling_ok: bool = False
    structured_output_ok: bool = False
    prefix_cache_verified: bool = False
    prefix_cache_note: str = ""
    reasoning_content_seen: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Startup gate: hard requirements only."""
        return (
            self.healthy
            and self.model_available
            and self.completion_ok
            and self.tool_calling_ok
            and self.structured_output_ok
        )


class InferenceHealthError(Exception):
    pass


async def verify_endpoint(inference: InferenceConfig) -> HealthReport:
    import httpx

    report = HealthReport()
    base = inference.base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. liveness
        try:
            response = await client.get(f"{root}/health")
            report.healthy = response.status_code == 200
        except httpx.HTTPError:
            report.healthy = False
        if not report.healthy:
            try:
                response = await client.get(f"{base}/models")
                report.healthy = response.status_code == 200
            except httpx.HTTPError as exc:
                report.errors.append(f"endpoint unreachable: {exc}")
                return report

        # 2. model availability
        try:
            response = await client.get(f"{base}/models")
            models = [m.get("id") for m in response.json().get("data", [])]
            report.model_available = inference.model in models
            if not report.model_available:
                report.errors.append(
                    f"model '{inference.model}' not served (available: {models})"
                )
        except (httpx.HTTPError, ValueError) as exc:
            report.errors.append(f"/models failed: {exc}")
            return report

        async def chat(messages, tools=None, max_tokens=256):
            payload = {
                "model": inference.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            response = await client.post(f"{base}/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()

        # 3. simple completion
        try:
            data = await chat([{"role": "user", "content": "Reply with the single word: ready"}])
            message = data["choices"][0]["message"]
            report.completion_ok = bool(
                (message.get("content") or "").strip()
                or message.get("reasoning_content")
            )
            report.usage_reported = bool(data.get("usage", {}).get("completion_tokens"))
            report.reasoning_content_seen = bool(message.get("reasoning_content"))
        except Exception as exc:
            report.errors.append(f"completion smoke test failed: {exc}")
            return report

        # 4. tool calling (qwen3_coder parser server-side)
        try:
            data = await chat(
                [{"role": "user",
                  "content": "Use the read_file tool to read the file 'README.md'. "
                             "Do not answer in text."}],
                tools=_TOOL_SMOKE_SCHEMA,
            )
            calls = data["choices"][0]["message"].get("tool_calls") or []
            ok = False
            for call in calls:
                function = call.get("function") or {}
                if function.get("name") == "read_file":
                    import json as _json
                    args = _json.loads(function.get("arguments") or "{}")
                    ok = "README" in str(args.get("path", ""))
            report.tool_calling_ok = ok
            if not ok:
                report.errors.append(
                    f"tool-calling smoke test produced no valid read_file call: {calls!r}"
                )
        except Exception as exc:
            report.errors.append(f"tool-calling smoke test failed: {exc}")

        # 5. structured output (normal generation + parse — the production path)
        try:
            data = await chat(
                [{"role": "user",
                  "content": 'Respond with ONLY this JSON in a ```json fence: {"status": "ok"}'}],
            )
            from .base import extract_json

            parsed = extract_json(data["choices"][0]["message"].get("content") or "")
            report.structured_output_ok = isinstance(parsed, dict) and "status" in parsed
            if not report.structured_output_ok:
                report.errors.append("structured-output smoke test: no parseable JSON")
        except Exception as exc:
            report.errors.append(f"structured-output smoke test failed: {exc}")

        # 6. prefix caching: option flags are not proof — observe metrics.
        try:
            before = await _prefix_cache_counters(client, root)
            shared = [{"role": "user", "content": _PREFIX_FILLER + " Reply: one"}]
            await chat(shared, max_tokens=8)
            shared2 = [{"role": "user", "content": _PREFIX_FILLER + " Reply: two"}]
            await chat(shared2, max_tokens=8)
            after = await _prefix_cache_counters(client, root)
            if before is None or after is None:
                report.prefix_cache_note = (
                    "vLLM /metrics not reachable from the harness; prefix caching "
                    "could not be verified end-to-end. Run "
                    "deploy/inference/healthcheck.sh on the serving host."
                )
                logger.warning(report.prefix_cache_note)
            elif after > before:
                report.prefix_cache_verified = True
            else:
                report.prefix_cache_note = (
                    "prefix cache metrics did not increase across two shared-prefix "
                    "requests — check --enable-prefix-caching on the server"
                )
                logger.warning(report.prefix_cache_note)
        except Exception as exc:
            report.prefix_cache_note = f"prefix cache verification errored: {exc}"
            logger.warning(report.prefix_cache_note)

    return report


async def _prefix_cache_counters(client, root: str) -> float | None:
    """Sum of prefix-cache hit counters from vLLM /metrics.

    Metric names vary across vLLM versions — matched by substring, never
    hardcoded to one release's naming.
    """
    try:
        response = await client.get(f"{root}/metrics")
        if response.status_code != 200:
            return None
    except Exception:
        return None
    total = 0.0
    seen = False
    for line in response.text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if "prefix_cache" in name and ("hit" in name or "queries" in name):
            match = re.search(r"\s([0-9.eE+-]+)$", line)
            if match:
                try:
                    total += float(match.group(1))
                    seen = True
                except ValueError:
                    continue
    return total if seen else None
