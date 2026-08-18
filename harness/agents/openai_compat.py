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
import hashlib
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

# Same chars-per-token heuristic the ContextBuilder budgets with.
CHARS_PER_TOKEN = 4
ELIDED_TOOL_RESULT = "[earlier tool result elided to fit the context window]"
ELIDED_ARGUMENTS = '{"_elided": true}'


def _elide_call_arguments(messages: list[dict], call_id: str | None) -> None:
    """Compact the arguments of a call whose result is already known.

    Once the tool has run, the outcome is what matters; the arguments are
    dead weight — and for write_file/edit_file they are the entire file
    body. Valid JSON is kept so the message still parses as a tool call.
    """
    if not call_id:
        return
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call.get("id") == call_id:
                function = call.get("function") or {}
                if function.get("arguments") != ELIDED_ARGUMENTS:
                    function["arguments"] = ELIDED_ARGUMENTS
                return


def _messages_size(messages: list[dict]) -> int:
    """Approximate character size of a chat history, tool calls included."""
    total = 0
    for message in messages:
        total += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            total += len(function.get("name") or "")
            total += len(function.get("arguments") or "")
    return total
_PERMANENT_STATUS = {401, 403, 404, 422}

# Shared clients keyed by (base_url, auth identity) — process-wide
# connection pooling. The auth identity is part of the key so a client
# built for one credential is never reused for another.
_CLIENTS: dict[tuple[str, str], Any] = {}

# Placeholder meaning "this endpoint needs no credential" (the default for
# a local vLLM started without --api-key).
NO_AUTH_PLACEHOLDER = "not-needed"


def auth_headers(api_key: str | None) -> dict[str, str]:
    """Bearer header for endpoints that require authentication."""
    if not api_key or api_key == NO_AUTH_PLACEHOLDER:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def shared_client(base_url: str, timeout_seconds: int, api_key: str | None = None):
    import httpx

    headers = auth_headers(api_key)
    # Identity, not the secret itself, keys the cache.
    key = (base_url, hashlib.sha256((api_key or "").encode()).hexdigest()[:16]
           if headers else "anon")
    client = _CLIENTS.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            headers=headers,
        )
        _CLIENTS[key] = client
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

    def __init__(self, inference: InferenceConfig,
                 gate: Optional[PrefixAffinityGate] = None,
                 resource_pools=None):
        self.inference = inference
        self.gate = gate or global_llm_gate(inference.concurrency.max_requests)
        # Host pools for agent-triggered builds/tests — the SAME limits the
        # harness's own verification draws from.
        self.resource_pools = resource_pools

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
        if not spec.model:
            # The invoker resolves the model via provider.for_role; there is
            # deliberately no inference-side fallback model to hide a broken
            # provider configuration behind.
            raise ValueError(
                f"no model resolved for role '{spec.role}': set provider.model "
                "or a provider.roles override"
            )
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
        pools = self.resource_pools
        executor = LocalToolExecutor(
            role=role,
            cwd=Path(spec.cwd),
            repo_root=Path(spec.repo_root),
            guard=guard,
            heavy_build_pool=getattr(pools, "heavy_build", None),
            heavy_test_pool=getattr(pools, "heavy_test", None),
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

    def _fit_to_window(self, messages: list[dict]) -> None:
        """Bound the accumulated tool loop to the configured input budget.

        The context budget shapes the INITIAL prompt, but each turn appends
        an assistant message plus a tool result, and the whole history is
        resent while output tokens stay reserved. One large read_file/grep
        result is enough to push a normal session past the server window.

        Oldest-first, tool results are replaced by a short notice — the
        message and its tool_call_id stay in place, because dropping a tool
        message whose assistant tool_calls remain would break the protocol.
        The system prompt and the task prompt are never touched: they are
        the assignment. Elision is permanent, so history cannot regrow.
        """
        budget_chars = self.inference.effective_input_budget() * CHARS_PER_TOKEN
        if _messages_size(messages) <= budget_chars:
            return
        # The results of the FINAL turn are the one part of the history the
        # model has not seen yet — eliding them would blind the loop to the
        # call it just made, and with a budget-sized initial prompt that
        # blindness repeats every turn until the repeat guard aborts. They
        # are exempt from both passes; over budget beats never seeing any
        # tool output.
        last_assistant = max(
            (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
            default=-1,
        )
        # Oldest completed turns first. A turn is elided as a PAIR: the tool
        # result and the arguments of the call that produced it. Those
        # arguments carry the whole file body for write_file/edit_file, so
        # eliding only the result leaves the larger half in the history.
        # The call keeps its id and name, so the assistant/tool pairing the
        # protocol requires still holds.
        for index, message in enumerate(messages):
            if _messages_size(messages) <= budget_chars:
                return
            if message.get("role") != "tool" or index > last_assistant:
                continue
            call_id = message.get("tool_call_id")
            if message.get("content") != ELIDED_TOOL_RESULT:
                message["content"] = ELIDED_TOOL_RESULT
            _elide_call_arguments(messages, call_id)
        # Still oversized with every tool result elided: drop older assistant
        # prose too (never the first two messages — system + assignment).
        for message in messages[2:]:
            if _messages_size(messages) <= budget_chars:
                return
            if message.get("role") == "assistant" and message.get("content"):
                message["content"] = ""
        if _messages_size(messages) > budget_chars:
            logger.warning(
                "conversation still exceeds the input budget after elision "
                "(%d chars); the request may be rejected by the server",
                _messages_size(messages),
            )

    async def _stream_chat(self, client, payload: dict):
        return await stream_chat_completion(client, payload)

    async def _chat(
        self, spec: ResolvedAgentRunSpec, messages: list[dict], tools: list[dict]
    ) -> dict:
        """One chat-completions call: gate slot held only for the request,
        transient failures retried in-session with backoff."""
        import httpx

        # Applied before every request (retries included) so a long tool
        # loop can never outgrow the window mid-session.
        self._fit_to_window(messages)
        sampling = spec.sampling or {}
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            # Non-streaming by default: internal agents display nothing, and
            # skipping SSE cuts host CPU/HTTP overhead. `inference.streaming`
            # switches the loop to SSE so the alternative is measurable
            # end-to-end, not just in the benchmark script.
            "stream": bool(self.inference.streaming),
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

        if payload["stream"]:
            payload["stream_options"] = {"include_usage": True}
        client = shared_client(spec.base_url or self.inference.base_url,
                               self.inference.request_timeout_seconds,
                               self.inference.api_key)
        delay = self.inference.transient_retry_base_delay
        last_error = "not attempted"
        for attempt in range(self.inference.transient_retries + 1):
            await self.gate.acquire(spec.prefix_group_key or None)
            try:
                if payload["stream"]:
                    message, status, body = await self._stream_chat(client, payload)
                else:
                    response = await client.post("/chat/completions", json=payload)
                    status, body = response.status_code, response.text[:500]
                    message = None
                    if status == 200:
                        # json() raising (proxy/overload returning 200 with an
                        # HTML or truncated body) is as transient as a 5xx —
                        # it must reach the in-session retry below, not abort
                        # the session and discard the tool-loop history.
                        data = response.json()
                        # Wrong-SHAPE 200 bodies (null, [], {"choices":[null]})
                        # would raise AttributeError/IndexError below, which
                        # the retry handler does not catch — same trust
                        # failure as bad syntax, so reject them as ValueError.
                        if not isinstance(data, dict):
                            raise ValueError(
                                f"response body is not an object: {body!r}")
                        # No `or {}` normalization BEFORE the checks: an
                        # empty choices list or a null message is the same
                        # trust failure — synthesizing an empty assistant
                        # message would end the session instead of retrying.
                        choices = data.get("choices")
                        if not isinstance(choices, list) or not choices:
                            raise ValueError(
                                f"response has no usable choices: {body!r}")
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            raise ValueError(
                                f"response choice is not an object: {body!r}")
                        raw_message = choice.get("message")
                        if not isinstance(raw_message, dict):
                            raise ValueError(
                                f"response message is not an object: {body!r}")
                        _validate_message_shape(raw_message, body)
                        message = dict(raw_message)
                        message["_usage"] = data.get("usage") or {}
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"transport error: {type(exc).__name__}: {exc}"
            else:
                if message is not None:
                    return message
                last_error = f"HTTP {status}: {body}"
                if status in _PERMANENT_STATUS:
                    raise PermanentHTTPError(last_error)
                if status not in _TRANSIENT_STATUS:
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


def _validate_message_shape(message: dict, context: str) -> None:
    """Reject a structurally invalid assistant message with ValueError so it
    reaches the in-session retry — the session loop's own field accesses
    would raise AttributeError/TypeError past it."""
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError(f"message content is not a string: {context!r}")
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        return
    if not isinstance(tool_calls, list):
        raise ValueError(f"message tool_calls is not a list: {context!r}")
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError(f"tool call is not an object: {context!r}")
        if call.get("id") is not None and not isinstance(call["id"], str):
            raise ValueError(f"tool call id is not a string: {context!r}")
        function = call.get("function")
        if function is None:
            continue
        if not isinstance(function, dict):
            raise ValueError(f"tool call function is not an object: {context!r}")
        for key in ("name", "arguments"):
            value = function.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"tool call function.{key} is not a string: {context!r}")


async def stream_chat_completion(client, payload: dict, url: str = "/chat/completions"):
    """Consume an SSE completion into the same message shape as the
    non-streaming path. Returns (message | None, status, body).

    Tool calls arrive as indexed fragments — name and id on the first
    delta for an index, `arguments` accumulating across later ones — so
    they are reassembled per index before the caller sees them.

    Module-level so the startup health gate can probe the exact same
    accumulator the agent loop runs on: a server that reports usage in
    ordinary JSON but omits it from SSE must fail startup, not silently
    record zero tokens for every streamed turn.
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    usage: dict = {}
    saw_chunk = False
    saw_done = False
    # Whether the server emitted hidden reasoning — a BOOLEAN, never the
    # text: reasoning is not persisted or forwarded anywhere. The health
    # probe needs it so a thinking model's response counts as a completion
    # on the streaming path exactly as it does on the JSON one.
    reasoning_seen = False
    async with client.stream("POST", url, json=payload) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", "replace")[:500]
            return None, response.status_code, body
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                # One damaged event means the stream can no longer be
                # trusted end-to-end: the lost fragment may be content or a
                # tool-call argument piece, and "skip and keep going" would
                # complete the turn with a silently partial response.
                raise ValueError(
                    f"malformed SSE data event ({exc}): {data[:200]!r}"
                )
            # Valid JSON of the wrong SHAPE (null, [], a string) is the same
            # trust failure as bad syntax — and letting AttributeError out
            # here would bypass the in-session retry that ValueError reaches.
            if not isinstance(chunk, dict):
                raise ValueError(f"SSE chunk is not an object: {data[:200]!r}")
            saw_chunk = True
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not isinstance(choices, list):
                raise ValueError(f"SSE chunk has non-list choices: {data[:200]!r}")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError(f"SSE choice is not an object: {data[:200]!r}")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise ValueError(f"SSE delta is not an object: {data[:200]!r}")
                delta_content = delta.get("content")
                if delta_content is not None and not isinstance(delta_content, str):
                    raise ValueError(
                        f"SSE delta content is not a string: {data[:200]!r}")
                if delta_content:
                    content_parts.append(delta_content)
                if delta.get("reasoning_content"):
                    reasoning_seen = True
                fragments = delta.get("tool_calls") or []
                if not isinstance(fragments, list):
                    raise ValueError(
                        f"SSE delta has non-list tool_calls: {data[:200]!r}")
                for fragment in fragments:
                    if not isinstance(fragment, dict):
                        raise ValueError(
                            f"SSE tool-call fragment is not an object: {data[:200]!r}")
                    index = fragment.get("index", 0)
                    if not isinstance(index, int) or isinstance(index, bool):
                        raise ValueError(
                            f"SSE fragment index is not an integer: {data[:200]!r}")
                    call = tool_calls.setdefault(
                        index,
                        {"id": "", "type": "function",
                         "function": {"name": "", "arguments": ""}},
                    )
                    fragment_id = fragment.get("id")
                    if fragment_id is not None and not isinstance(fragment_id, str):
                        raise ValueError(
                            f"SSE fragment id is not a string: {data[:200]!r}")
                    if fragment_id:
                        call["id"] = fragment_id
                    function = fragment.get("function") or {}
                    if not isinstance(function, dict):
                        raise ValueError(
                            f"SSE tool-call function is not an object: {data[:200]!r}")
                    for key in ("name", "arguments"):
                        value = function.get(key)
                        if value is not None and not isinstance(value, str):
                            raise ValueError(
                                f"SSE function.{key} is not a string: {data[:200]!r}")
                    if function.get("name"):
                        call["function"]["name"] = function["name"]
                    if function.get("arguments"):
                        call["function"]["arguments"] += function["arguments"]
    if not (saw_chunk and saw_done):
        # An HTML body, a stream of unparseable events, or a stream cut off
        # before its [DONE] sentinel. Accepting it would complete a tool
        # turn with an empty/partial response and bypass the in-session
        # retry that exists exactly for transient transport damage.
        raise ValueError(
            "malformed or truncated SSE stream "
            f"(parsed_chunks={saw_chunk}, done_sentinel={saw_done})"
        )
    message: dict = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    message["_usage"] = usage
    message["_reasoning_seen"] = reasoning_seen
    return message, 200, ""
