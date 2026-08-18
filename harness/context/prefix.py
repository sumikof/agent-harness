"""Prompt-prefix stability helpers for vLLM prefix caching.

Prefix caching only pays off when many agent sessions share a long,
byte-identical token prefix. Two things make that happen:

1. Stable serialization — JSON embedded in prompts is rendered
   canonically (sorted keys, fixed separators), so the same logical
   content always produces the same bytes.
2. PrefixGroupKey — a deterministic fingerprint of everything that makes
   up a session's stable prefix (model + role profile + project context +
   tool schema + shared prompt text). Sessions with equal keys share a
   cacheable prefix; the scheduler uses the key as a secondary dispatch
   criterion so such sessions land on the server close together.

Nothing volatile (timestamps, run ids, attempt numbers, diffs) may enter
the key or the stable prefix — that lives in the prompt tail.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any, indent: int | None = None) -> str:
    """Deterministic JSON: sorted keys, fixed separators, stable unicode.

    Identical logical content always serializes to identical bytes, so a
    serialization difference can never break a shared prompt prefix.
    """
    if indent is None:
        return json.dumps(
            data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
        )
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=indent, default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prefix_group_key(
    *,
    model: str,
    agent_profile_hash: str,
    project_context_hash: str,
    tool_schema_hash: str,
    common_prompt_hash: str,
) -> str:
    """Fingerprint of a session's cacheable stable prefix.

    Equal keys => the sessions' first N thousand tokens are identical and
    a second dispatch can reuse the first one's KV prefix. Different
    roles/projects/tool catalogs produce different keys by construction.
    """
    material = canonical_json(
        {
            "model": model,
            "agent_profile_hash": agent_profile_hash,
            "project_context_hash": project_context_hash,
            "tool_schema_hash": tool_schema_hash,
            "common_prompt_hash": common_prompt_hash,
        }
    )
    return sha256_hex(material)


# Names that must never appear inside the stable prefix sections. Used by
# tests to guard against regressions that would silently kill cache reuse.
VOLATILE_MARKERS = (
    "attempt number",
    "run id",
    "agent_run_id",
    "timestamp",
)
