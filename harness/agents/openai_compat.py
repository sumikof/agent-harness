"""Local OpenAI-compatible provider (vLLM serving Qwen3.6-27B-FP8).

Everything HTTP/OpenAI-specific lives here — the core orchestrator only
sees the AgentRunner protocol. Design points for DGX Spark throughput:

- ONE shared httpx.AsyncClient per endpoint (connection pooling /
  keep-alive); never a client per AgentRun.
- ONE process-wide PrefixAffinityGate caps in-flight requests (default
  16 = the serving --max-num-seqs baseline) and dispatches queued
  requests with prefix affinity. The slot is held ONLY for the HTTP
  request — tool execution releases it so other agents can infer.
- Non-streaming by default (config `inference.streaming`): internal
  agents don't display tokens, and skipping SSE reduces host overhead.
- Transient failures (429 / 5xx / transport) are retried IN-SESSION with
  exponential backoff, preserving the message history. Overload is
  backpressure, never a task failure and never an extra task attempt.
- Thinking mode: vLLM's reasoning parser returns `reasoning_content`
  separately; it is used within the session's own turns only and is
  NEVER persisted to artifacts or forwarded to other agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..config import InferenceConfig
from ..orchestrator.resources import PrefixAffinityGate, global_llm_gate
from ..orchestrator.state_machine import Role
from ..security.hooks import RepeatActionGuard
from .base import AgentResult, BaseAgentRunner, FailureKind, extract_json
from .local_tools import LocalToolExecutor, tool_schema_hash, tool_schemas_for
from .profile import AgentCapabilities, ResolvedAgentRunSpec

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_PERMANENT_STATUS = {401, 403, 404, 422}

# Shared clients keyed by base_url — process-wide connection pooling.
_CLIENTS: dict[str, Any] = {}


def shared_client(base_url: str, timeout_seconds: int):
    import httpx

    client = _CLIENTS.get(base_url)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        _CLIENTS[base_url] = client
    return client


class TransientHTTPError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class PermanentHTTPError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class LocalOpenAICompatibleAgentRunner(BaseAgentRunner):
    """Agentic tool loop over a local OpenAI-compatible chat endpoint."""

    provider_name = "openai-compatible"

    def __init__(self, inference: InferenceConfig, gate: Optional[PrefixAffinityGate] = None):
        self.inference = inference
        self.gate = gate or global_llm_gate(inference.concurrency.max_requests)

    def capabilities(self) -> AgentCapabilities:
        # Declared here; VERIFIED against the live endpoint by
        # harness.agents.health.verify_endpoint() before any project runs.
        return AgentCapabilities(
            structured_output=True,   # JSON extraction + one guided retry
            session_resume=False,     # fresh sessions by design
            tool_permissions=True,    # harness-owned tool loop enforces them
            pre_tool_hook=True,       # security + repeat guard run pre-execution
            post_tool_hook=False,
            cancellation=True,        # cooperative: HTTP request cancellation
            hard_termination=False,
            usage_reporting=True,     # vLLM returns usage on completions
        )

    async def resolve(self, request) -> ResolvedAgentRunSpec:
        spec = await super().resolve(request)
        role = Role(spec.role)
        sampling = self.inference.sampling_for_role(role.value)
        spec.base_url = self.inference.base_url
        spec.model = spec.model or self.inference.model
        spec.sampling = sampling.model_dump()
        spec.max_output_tokens = self.inference.max_output_tokens
        spec.tool_schema_hash = tool_schema_hash(role)
        return spec

    # ------------------------------------------------------------------

    async def run(self, spec: ResolvedAgentRunSpec) -> AgentResult:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            return AgentResult(
                status="FAILED",
                error=f"httpx is not installed: {exc}. pip install httpx",
                failure_kind=FailureKind.PERMANENT,
            )

        role = Role(spec.role)
        guard: Optional[RepeatActionGuard] = None
        if spec.repeat_guard is not None and spec.repeat_guard.enabled:
            guard = RepeatActionGuard(
                warn_after=spec.repeat_guard.warn_after,
                abort_after=spec.repeat_guard.abort_after,
                exempt_tools=spec.repeat_guard.exempt_tools,
            )
        executor = LocalToolExecutor(
            role=role,
            cwd=Path(spec.cwd),
            repo_root=Path(spec.repo_root),
            guard=guard,
        )
        tools = tool_schemas_for(role)

        messages: list[dict] = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": spec.prompt},
        ]
        result = AgentResult(status="FAILED")
        usage_totals = {"input_tokens": 0, "output_tokens": 0}
        text_parts: list[str] = []

        async def session() -> None:
            for turn in range(spec.max_turns):
                message = await self._chat(spec, messages, tools)
                result.num_turns = turn + 1
                usage = message.pop("_usage", None) or {}
                usage_totals["input_tokens"] += usage.get("prompt_tokens", 0) or 0
                usage_totals["output_tokens"] += usage.get("completion_tokens", 0) or 0

                tool_calls = message.get("tool_calls") or []
                content = message.get("content") or ""
                if content:
                    text_parts.append(content)
                # Hidden reasoning (`reasoning_content`) is deliberately NOT
                # appended back and never persisted — Fresh Agent + Structured
                # Handoff stays intact.
                assistant_msg: dict = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                if not tool_calls:
                    result.status = "COMPLETED"
                    return
                for call in tool_calls:
                    function = call.get("function") or {}
                    name = function.get("name", "")
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError("arguments must be an object")
                    except (json.JSONDecodeError, ValueError) as exc:
                        tool_output = f"TOOL ERROR: unparseable arguments: {exc}"
                    else:
                        tool_output = await executor.execute(name, arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": tool_output,
                    })
                    if executor.loop_detected:
                        result.status = "FAILED"
                        result.error = "LOOP_DETECTED: identical tool call repeated beyond abort threshold"
                        return
            result.status = "FAILED"
            result.error = f"agent exceeded max_turns={spec.max_turns} without a final answer"

        try:
            if spec.timeout_seconds:
                await asyncio.wait_for(session(), timeout=spec.timeout_seconds)
            else:
                await session()
        except asyncio.TimeoutError:
            result.status = "FAILED"
            result.error = f"agent run exceeded timeout of {spec.timeout_seconds}s"
            result.failure_kind = FailureKind.TRANSIENT
        except PermanentHTTPError as exc:
            result.status = "FAILED"
            result.error = exc.detail
            result.failure_kind = FailureKind.PERMANENT
        except TransientHTTPError as exc:
            result.status = "FAILED"
            result.error = exc.detail
            result.failure_kind = FailureKind.TRANSIENT
        except Exception as exc:
            result.status = "FAILED"
            result.error = f"{type(exc).__name__}: {exc}"
            result.failure_kind = FailureKind.TRANSIENT

        result.output_text = "\n".join(text_parts)
        result.token_usage = dict(usage_totals)
        result.cost_usd = 0.0  # local inference: no per-token billing
        if result.status == "COMPLETED":
            result.structured_output = extract_json(result.output_text)
        if guard is not None:
            result.loop_detected = result.loop_detected or guard.loop_detected
            result.loop_warnings = list(guard.warnings)
        result.telemetry = {
            "tool_calls": executor.tool_calls,
            "turns": result.num_turns,
            "input_tokens": usage_totals["input_tokens"],
            "output_tokens": usage_totals["output_tokens"],
            "llm_queue_depth": self.gate.queue_depth,
            "llm_in_flight": self.gate.in_flight,
        }
        return result

    # ------------------------------------------------------------------

    async def _chat(
        self, spec: ResolvedAgentRunSpec, messages: list[dict], tools: list[dict]
    ) -> dict:
        """One chat-completions call: gate slot held only for the request,
        transient failures retried in-session with backoff."""
        import httpx

        sampling = spec.sampling or {}
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            # The internal loop is non-streaming: agents don't display tokens
            # and skipping SSE reduces host CPU/HTTP overhead (config
            # `inference.streaming` exists for benchmarking the alternative).
            "stream": False,
            "temperature": sampling.get("temperature", 0.6),
            "top_p": sampling.get("top_p", 0.95),
            "max_tokens": spec.max_output_tokens or self.inference.max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        # vLLM-specific samplers travel in extra_body-compatible top level.
        for key in ("top_k", "min_p", "repetition_penalty", "presence_penalty"):
            if key in sampling:
                payload[key] = sampling[key]

        client = shared_client(spec.base_url or self.inference.base_url,
                               self.inference.request_timeout_seconds)
        delay = self.inference.transient_retry_base_delay
        last_error = "not attempted"
        for attempt in range(self.inference.transient_retries + 1):
            await self.gate.acquire(spec.prefix_group_key or None)
            try:
                response = await client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    message = dict(choice.get("message") or {})
                    message["_usage"] = data.get("usage") or {}
                    return message
                body = response.text[:500]
                last_error = f"HTTP {response.status_code}: {body}"
                if response.status_code in _PERMANENT_STATUS:
                    raise PermanentHTTPError(last_error)
                if response.status_code not in _TRANSIENT_STATUS:
                    raise PermanentHTTPError(last_error)
            finally:
                self.gate.release(spec.prefix_group_key or None)
            if attempt < self.inference.transient_retries:
                logger.warning(
                    "%s: transient LLM failure (%s); retrying in %.1fs",
                    spec.role, last_error, delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
        raise TransientHTTPError(f"LLM request failed after retries: {last_error}")
