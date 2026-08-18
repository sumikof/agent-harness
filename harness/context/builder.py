"""Context Builder.

Conversation history is never long-term state: the harness assembles a
fresh prompt for every agent session from the three context layers,
selecting only what the role needs.

Prompt layout is prefix-cache-aware. Sections are ordered strictly
stable -> dynamic so that sessions of the same role/project share the
longest possible identical token prefix on the vLLM side:

    (system prompt: stable global instructions + role profile)
    1. harness_rules       stable across the whole deployment
    2. project_context     stable across the project
    ----------------------------- stable prefix boundary -----------
    3. task_context        task-specific
    4. attempt_context     attempt-specific (failures, feedback)
    5. extra_context       invocation-specific
    6. volatile_metadata   working dir, attempt number, branch, ...
    7. assignment          stable text, but placed last so the final
                           instruction sits next to the answer

Nothing volatile (timestamps, ids, diffs, working directories) may
appear in sections 1-2. Embedded JSON uses canonical serialization so
identical logical content always produces identical bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from ..orchestrator.state_machine import Role
from .attempt_context import AttemptContext
from .prefix import canonical_json, prefix_group_key, sha256_hex
from .project_context import ProjectContext
from .task_context import TaskContext

# Stable harness rules every agent receives, identical bytes for every
# session in a deployment. Keep free of anything project- or run-specific.
HARNESS_RULES = """## Harness rules
- You are one role in a deterministic multi-agent harness. The harness — not you — decides what runs next.
- Work only inside your current working directory (an isolated git worktree for this task).
- Never run git commit / merge / rebase / push / reset / worktree commands; the harness owns the git lifecycle.
- Communicate results ONLY through the required JSON artifact described in your role instructions.
- Do not rely on conversation memory: everything you need is in this prompt; everything you produce must be in your artifact.
"""

# Rough chars-per-token for budget accounting. Precise tokenization is a
# provider concern; the budget only needs to keep prompts inside the
# serving profile with headroom for reasoning + tool calls + output.
CHARS_PER_TOKEN = 4
TRUNCATION_NOTICE = "\n... (section truncated by context budget; full content in artifacts)\n"
# Preferred minimum a truncated section keeps, capped by what the budget
# can actually hold.
SECTION_FLOOR_CHARS = 2000

# Dynamic sections eligible for budget truncation, largest first. Stable
# sections are never truncated — cutting them would both lose rules and
# fracture the shared prefix.
_TRUNCATABLE = ("extra_context", "attempt_context", "task_context")


class ContextBudgetError(ValueError):
    """Stable prompt content alone exceeds the configured input budget."""


class ContextBuilder:
    def __init__(self, prompts_dir: Path, input_budget_tokens: int | None = None):
        self.prompts_dir = prompts_dir
        self.input_budget_tokens = input_budget_tokens

    def system_prompt(self, prompt_file: str) -> str:
        path = self.prompts_dir / prompt_file
        return path.read_text(encoding="utf-8")

    def prompt_template_hash(self, prompt_file: str) -> str:
        return hashlib.sha256(self.system_prompt(prompt_file).encode("utf-8")).hexdigest()

    # -- section assembly ---------------------------------------------------

    def build_sections(
        self,
        role: Role,
        project: ProjectContext,
        task: Optional[TaskContext] = None,
        attempt: Optional[AttemptContext] = None,
        extra: str = "",
        volatile: Optional[dict] = None,
        system_prompt: str = "",
    ) -> list[tuple[str, str]]:
        """The named context sections one session receives, in prompt order.

        Named so each section can be persisted individually in the
        ContextManifest; build_prompt() joins exactly these texts.

        `system_prompt` is not returned — the provider sends it as its own
        message — but its size counts against the budget, because the
        server sees one request. Trimming to a budget that ignored it would
        put every large prompt over the real limit before a single tool
        result was appended.
        """
        sections: list[tuple[str, str]] = [
            ("harness_rules", HARNESS_RULES),
            ("project_context", project.render()),
        ]

        if task is not None:
            # The Analyst produces the brief, so it doesn't receive one;
            # the Reviewer judges the diff on its own terms but needs all artifacts.
            include_brief = role != Role.ANALYST
            sections.append(("task_context", task.render(include_brief=include_brief)))

        if attempt is not None and attempt.has_content():
            sections.append(("attempt_context", attempt.render()))

        if extra:
            sections.append(("extra_context", extra))

        if volatile:
            sections.append(("volatile_metadata", self._render_volatile(volatile)))

        sections.append(("assignment", self._task_instruction(role)))
        return self._enforce_budget(sections, reserved_chars=len(system_prompt))

    def build_prompt(
        self,
        role: Role,
        project: ProjectContext,
        task: Optional[TaskContext] = None,
        attempt: Optional[AttemptContext] = None,
        extra: str = "",
        volatile: Optional[dict] = None,
        system_prompt: str = "",
    ) -> str:
        """Assemble Project + Task + (needed) Attempt context for one session."""
        return "\n\n".join(
            text for _, text in self.build_sections(
                role, project, task, attempt, extra, volatile, system_prompt)
        )

    # -- prefix cache -------------------------------------------------------

    STABLE_SECTIONS = ("harness_rules", "project_context")

    def stable_prefix_text(self, role: Role, project: ProjectContext) -> str:
        """The user-prompt prefix shared by every session of this project.

        Volatile inputs cannot reach it by construction: it is built from
        the static rules and the project context only.
        """
        return "\n\n".join((HARNESS_RULES, project.render()))

    def project_context_hash(self, project: ProjectContext) -> str:
        return sha256_hex(project.render())

    def compute_prefix_group_key(
        self,
        *,
        model: str,
        agent_profile_hash: str,
        project: ProjectContext,
        role: Role,
        prompt_file: str,
        tool_schema_hash: str = "",
    ) -> str:
        """PrefixGroupKey for one prospective session: sessions with equal
        keys share their stable prompt prefix byte-for-byte."""
        common = sha256_hex(
            self.system_prompt(prompt_file) + "\n\n" + self.stable_prefix_text(role, project)
        )
        return prefix_group_key(
            model=model,
            agent_profile_hash=agent_profile_hash,
            project_context_hash=self.project_context_hash(project),
            tool_schema_hash=tool_schema_hash,
            common_prompt_hash=common,
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _render_volatile(volatile: dict) -> str:
        """Request-specific metadata, rendered canonically, always at the
        prompt tail — it must never break the shared stable prefix."""
        lines = ["## Session metadata (volatile)"]
        for key in sorted(volatile):
            lines.append(f"- {key}: {volatile[key]}")
        return "\n".join(lines)

    def _enforce_budget(
        self, sections: list[tuple[str, str]], reserved_chars: int = 0
    ) -> list[tuple[str, str]]:
        if not self.input_budget_tokens:
            return sections
        # The role's system prompt is part of the same request; it is never
        # truncated (it defines the role), so it is reserved off the top.
        budget_chars = max(
            self.input_budget_tokens * CHARS_PER_TOKEN - reserved_chars, 0
        )
        # The joiner between sections is part of what the runner sends, so
        # it counts too — otherwise a budget met exactly is still exceeded.
        joiner_chars = 2 * max(len(sections) - 1, 0)
        total = sum(len(text) for _, text in sections) + joiner_chars
        if total <= budget_chars:
            return sections
        # Trim dynamic sections only, in fixed order, until within budget.
        trimmed = dict(sections)
        for name in _TRUNCATABLE:
            if total <= budget_chars:
                break
            text = trimmed.get(name)
            if not text:
                continue
            excess = total - budget_chars
            # Keep a readable slice, but never one the budget cannot hold:
            # a section floor larger than the budget would leave the request
            # over the limit, which is worse than a shorter section.
            floor = min(SECTION_FLOOR_CHARS, max(budget_chars // 4, 0))
            keep = max(len(text) - excess - len(TRUNCATION_NOTICE), floor)
            if keep >= len(text):
                continue
            head = text[: keep * 2 // 3]
            tail = text[-(keep - len(head)):] if keep > len(head) else ""
            trimmed[name] = head + TRUNCATION_NOTICE + tail
            total -= len(text) - len(trimmed[name])
        if total > budget_chars:
            # The preferred floor kept dynamic sections readable; before
            # declaring overflow, spend that floor too — a squeezed section
            # is recoverable (full content lives in artifacts), a rejected
            # project is not.
            for name in _TRUNCATABLE:
                if total <= budget_chars:
                    break
                text = trimmed.get(name)
                if not text:
                    continue
                replacement = TRUNCATION_NOTICE.strip()
                if len(replacement) < len(text):
                    total -= len(text) - len(replacement)
                    trimmed[name] = replacement
        if total > budget_chars:
            # Even the notices don't fit: drop dynamic sections outright —
            # an absent section is recoverable via artifacts, a rejected
            # project is not. Only then can a remaining overflow be blamed
            # on stable content.
            for name in _TRUNCATABLE:
                if total <= budget_chars:
                    break
                text = trimmed.get(name)
                if text:
                    total -= len(text)
                    trimmed[name] = ""
        if total > budget_chars:
            # Only stable, untrimmable content is left (system prompt, rules,
            # project context, assignment). The provider never truncates
            # those either, so every request for this project would be
            # rejected for context length — fail once, loudly, with the
            # numbers, instead of failing on every dispatch.
            raise ContextBudgetError(
                f"stable prompt content ({total} chars + {reserved_chars} "
                f"reserved for the system prompt) exceeds the input budget "
                f"({self.input_budget_tokens} tokens = "
                f"{self.input_budget_tokens * CHARS_PER_TOKEN} chars); "
                "shorten the project context / role prompt or select a "
                "larger context_profile"
            )
        return [(name, trimmed[name]) for name, _ in sections]

    def _task_instruction(self, role: Role) -> str:
        instructions = {
            Role.PLANNER: "Analyze the repository and produce the project plan JSON now.",
            Role.ANALYST: "Investigate the repository for this task and produce the task brief JSON now.",
            Role.DEVELOPER: "Implement the task following the brief. When done, produce the implementation JSON.",
            Role.TESTER: "Review test coverage for this change, add missing tests, and produce the test report JSON.",
            Role.REVIEWER: "Review the change against the acceptance criteria and produce the review JSON.",
            Role.DIAGNOSTICIAN: "Analyze why this task keeps failing and produce the diagnosis JSON.",
        }
        return f"## Your assignment\n{instructions[role]}"


def render_plan_for_replan(existing_tasks: list[dict]) -> str:
    """Extra context handed to the Planner when re-planning."""
    return (
        "## Existing plan state\n"
        "The current plan could not proceed. Completed tasks must NOT be re-planned; "
        "revise only the remaining work.\n"
        "```json\n" + canonical_json(existing_tasks, indent=2) + "\n```"
    )
