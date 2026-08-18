"""Inference endpoint health checks and capability verification.

Capabilities are never assumed: before the harness starts a project on
the local OpenAI-compatible provider, the endpoint must pass

    1. liveness            GET /health (vLLM) or GET /v1/models
    2. model availability  every model a local role dispatches is served
    3. simple completion   a trivial prompt returns text + usage
    4. tool calling        the model emits a correct read_file tool call
                           (qwen3_coder parser on the server side)
    5. structured output   the model returns parseable JSON on request

Probes 3-5 run against EVERY effective role model, not just the first.
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
from .openai_compat import auth_headers, stream_chat_completion

logger = logging.getLogger(__name__)

# Stand-in recorded when a streamed response carried reasoning deltas. The
# reasoning text itself is never kept — only the fact that the parser fired.
REASONING_SEEN_MARKER = "<reasoning emitted>"

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

# Capability probes run with the reasoning parser active: a thinking model
# spends tokens on reasoning_content BEFORE the tool call / JSON it was
# asked for. A tight cap (e.g. 256) truncates mid-thought and fails a
# healthy endpoint. Startup-only, so the larger budget costs nothing.
PROBE_MAX_TOKENS = 2048

# Long shared prefix for the cache smoke test (needs to exceed the
# server's prefix-match block size comfortably).
_PREFIX_FILLER = ("The harness verifies serving features before use. " * 400).strip()


@dataclass
class ModelReport:
    """Capability probe results for ONE served model."""

    model: str
    available: bool = False
    completion_ok: bool = False
    usage_reported: bool = False
    tool_calling_ok: bool = False
    structured_output_ok: bool = False
    reasoning_content_seen: bool = False

    @property
    def ok(self) -> bool:
        # usage_reported is required, not informational: the runner declares
        # a usage_reporting capability and records absent counts as zero, so
        # an endpoint without it would silently corrupt token telemetry and
        # every budget derived from it.
        return (
            self.available
            and self.completion_ok
            and self.usage_reported
            and self.tool_calling_ok
            and self.structured_output_ok
        )


@dataclass
class HealthReport:
    healthy: bool = False
    # One entry per model a local role will actually dispatch. Capabilities
    # are probed on EVERY one of them: a second served model that lacks the
    # tool parser or usable structured output must not pass startup and
    # surface only once its role starts a task.
    models: dict = field(default_factory=dict)      # model -> ModelReport
    prefix_cache_verified: bool = False
    prefix_cache_note: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def models_checked(self) -> dict:
        return {name: report.available for name, report in self.models.items()}

    @property
    def model_available(self) -> bool:
        return bool(self.models) and all(r.available for r in self.models.values())

    def _all(self, attribute: str) -> bool:
        return bool(self.models) and all(
            getattr(r, attribute) for r in self.models.values()
        )

    # Aggregates across every probed model.
    @property
    def completion_ok(self) -> bool:
        return self._all("completion_ok")

    @property
    def usage_reported(self) -> bool:
        return self._all("usage_reported")

    @property
    def tool_calling_ok(self) -> bool:
        return self._all("tool_calling_ok")

    @property
    def structured_output_ok(self) -> bool:
        return self._all("structured_output_ok")

    @property
    def reasoning_content_seen(self) -> bool:
        return any(r.reasoning_content_seen for r in self.models.values())

    @property
    def ok(self) -> bool:
        """Startup gate: every effective role model must pass."""
        return self.healthy and bool(self.models) and all(
            r.ok for r in self.models.values()
        )


class InferenceHealthError(Exception):
    pass


async def verify_endpoint(
    inference: InferenceConfig, models: list[str] | None = None
) -> HealthReport:
    """Verify the endpoint against the models the harness will really use.

    `models` is the set of effective role models (see
    `harness.main.local_role_models`); it defaults to the inference default
    alone. Checking only the default would pass a configuration whose roles
    dispatch a model the server does not serve — or reject a valid one whose
    default is simply unused.
    """
    import httpx

    report = HealthReport()
    base = inference.base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    required = list(dict.fromkeys(models or [inference.model]))

    async with httpx.AsyncClient(
        timeout=60.0, headers=auth_headers(inference.api_key)
    ) as client:
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

        # 2. model availability — for EVERY model a local role dispatches
        try:
            response = await client.get(f"{base}/models")
            served = [m.get("id") for m in response.json().get("data", [])]
            report.models = {name: ModelReport(model=name, available=name in served)
                             for name in required}
            missing = [name for name in required if name not in served]
            if missing:
                report.errors.append(
                    f"model(s) {missing} not served (available: {served})"
                )
        except (httpx.HTTPError, ValueError) as exc:
            report.errors.append(f"/models failed: {exc}")
            return report

        async def chat(messages, model, tools=None, max_tokens=PROBE_MAX_TOKENS):
            """One probe request, issued through the SAME response path the
            agents will use.

            Probing only JSON completions would pass an endpoint that
            reports usage in ordinary responses while omitting it from SSE;
            the streaming loop would then record zero tokens on every turn
            and no startup gate would have caught it. So when
            `inference.streaming` is on, the probe streams and is
            reassembled by the production accumulator.
            """
            streaming = bool(inference.streaming)
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": streaming,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            if streaming:
                payload["stream_options"] = {"include_usage": True}
                message, status, body = await stream_chat_completion(
                    client, payload, url=f"{base}/chat/completions"
                )
                if message is None:
                    raise InferenceHealthError(f"HTTP {status}: {body}")
                usage = message.pop("_usage", None) or {}
                if message.pop("_reasoning_seen", False):
                    # A flag, not the text: enough for the probe to see that
                    # the reasoning parser is active without the harness ever
                    # holding hidden reasoning.
                    message["reasoning_content"] = REASONING_SEEN_MARKER
                return {"choices": [{"message": message}], "usage": usage}
            response = await client.post(f"{base}/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()

        # 3-5. capability probes, per model. A role dispatching a model that
        # cannot tool-call or emit parseable JSON must fail startup, not the
        # first task that happens to use it.
        for name, model_report in report.models.items():
            if not model_report.available:
                continue

            # 3. simple completion
            try:
                data = await chat(
                    [{"role": "user", "content": "Reply with the single word: ready"}],
                    name,
                )
                message = data["choices"][0]["message"]
                model_report.completion_ok = bool(
                    (message.get("content") or "").strip()
                    or message.get("reasoning_content")
                )
                # BOTH counts, not just one: a partial usage object leaves
                # every missing prompt count recorded as zero, so input
                # telemetry and the budgets derived from it stay corrupt.
                usage = data.get("usage") or {}
                model_report.usage_reported = all(
                    isinstance(usage.get(key), int) and usage[key] > 0
                    for key in ("prompt_tokens", "completion_tokens")
                )
                model_report.reasoning_content_seen = bool(
                    message.get("reasoning_content"))
            except Exception as exc:
                report.errors.append(f"[{name}] completion smoke test failed: {exc}")
                continue
            if not model_report.usage_reported:
                report.errors.append(
                    f"[{name}] endpoint did not report both prompt_tokens and "
                    "completion_tokens; token telemetry and every budget derived "
                    "from it would be wrong"
                )

            # 4. tool calling (qwen3_coder parser server-side)
            try:
                data = await chat(
                    [{"role": "user",
                      "content": "Use the read_file tool to read the file 'README.md'. "
                                 "Do not answer in text."}],
                    name,
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
                model_report.tool_calling_ok = ok
                if not ok:
                    report.errors.append(
                        f"[{name}] tool-calling smoke test produced no valid "
                        f"read_file call: {calls!r}"
                    )
            except Exception as exc:
                report.errors.append(f"[{name}] tool-calling smoke test failed: {exc}")

            # 5. structured output (normal generation + parse — production path)
            try:
                data = await chat(
                    [{"role": "user",
                      "content": 'Respond with ONLY this JSON in a ```json fence: '
                                 '{"status": "ok"}'}],
                    name,
                )
                from .base import extract_json

                parsed = extract_json(data["choices"][0]["message"].get("content") or "")
                model_report.structured_output_ok = (
                    isinstance(parsed, dict) and "status" in parsed)
                if not model_report.structured_output_ok:
                    report.errors.append(
                        f"[{name}] structured-output smoke test: no parseable JSON")
            except Exception as exc:
                report.errors.append(
                    f"[{name}] structured-output smoke test failed: {exc}")

        # 6. prefix caching: option flags are not proof — observe metrics.
        try:
            before = await _prefix_cache_counters(client, root)
            probe = next((n for n, r in report.models.items() if r.available),
                         required[0])
            shared = [{"role": "user", "content": _PREFIX_FILLER + " Reply: one"}]
            await chat(shared, probe, max_tokens=8)
            shared2 = [{"role": "user", "content": _PREFIX_FILLER + " Reply: two"}]
            await chat(shared2, probe, max_tokens=8)
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
    """Sum of prefix-cache HIT counters from vLLM /metrics.

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
        # HIT counters only. A queries/lookups counter advances for the probe
        # requests regardless of whether anything was cached, which would
        # report an ineffective cache as verified.
        if "prefix_cache" in name and "hit" in name:
            match = re.search(r"\s([0-9.eE+-]+)$", line)
            if match:
                try:
                    total += float(match.group(1))
                    seen = True
                except ValueError:
                    continue
    return total if seen else None
