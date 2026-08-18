"""Local tool schemas + executor for the OpenAI-compatible provider.

The Claude Agent SDK brings its own tool loop; a raw OpenAI-compatible
endpoint (vLLM) does not — the harness owns the loop, the tool catalog,
and the enforcement. Three properties matter:

1. STABLE SCHEMAS. The tool catalog for a given role profile is a fixed
   list in a fixed order with canonical serialization: identical bytes on
   every request, so the tool block never breaks the vLLM prefix cache.
2. ENFORCED PERMISSIONS. Every call passes the same security policy the
   Claude adapter enforces via its PreToolUse hook (forbidden commands,
   write-path checks, read-only Bash for non-developers). Prompts alone
   never grant a tool.
3. BOUNDED OUTPUT. Large tool results are truncated with a notice —
   the full content stays on disk, not in context.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

# Single source for the tool-result cap: the prompt-budget reserve in
# config depends on it, so it lives there.
from ..config import MAX_TOOL_OUTPUT_CHARS
from ..concurrency import run_thread_uninterruptible
from ..process import run_command
from ..context.prefix import canonical_json, sha256_hex
from ..orchestrator.state_machine import Role
from ..security.commands import check_command, find_write_hint
from ..security.hooks import RepeatActionGuard
from ..security.permissions import check_write_path

# -- schemas (FIXED order; never reorder — it would invalidate cached
#    prefixes and change profile hashes) -----------------------------------

_STRING = {"type": "string"}
_INT = {"type": "integer"}

READ_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository. Returns the content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _STRING,
                    "offset": {**_INT, "description": "1-based first line to read"},
                    "limit": {**_INT, "description": "maximum number of lines"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {"path": _STRING},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern (e.g. 'src/**/*.py').",
            "parameters": {
                "type": "object",
                "properties": {"pattern": _STRING},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with a regular expression. Returns matching lines as path:line:text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": _STRING,
                    "path": {**_STRING, "description": "directory or file to search (default: repository root)"},
                    "glob": {**_STRING, "description": "only search files matching this glob"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the repository working directory. Returns exit code and output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": _STRING,
                    "timeout_seconds": _INT,
                },
                "required": ["command"],
            },
        },
    },
]

WRITE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {"path": _STRING, "content": _STRING},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file. old_string must match exactly and be unique unless replace_all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _STRING,
                    "old_string": _STRING,
                    "new_string": _STRING,
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
]

_WRITER_ROLES = {Role.DEVELOPER, Role.TESTER}

# Host-heavy shell commands. An agent running `pytest` or `mvn` consumes the
# same CPU/RAM as the harness's own verification step, so it must draw from
# the same bounded pools — otherwise `max_parallel_agent_runs` expensive
# builds can run at once and exhaust the machine that is also serving the
# model. Substring matching over the normalized command line, so chained
# forms (`cd x && pytest -q`) are caught too. A heuristic by nature:
# over-matching only serializes more work, under-matching is the status quo.
HEAVY_TEST_MARKERS = (
    "pytest", "py.test", "unittest", "tox", "nosetests",
    "jest", "vitest", "mocha", "npm test", "yarn test", "pnpm test",
    "go test", "cargo test", "mvn test", "mvn verify", "gradle test",
    "gradlew test", "rspec", "phpunit", "dotnet test", "ctest",
)
HEAVY_BUILD_MARKERS = (
    "mvn ", "gradle ", "gradlew ", "make ", "cmake", "cargo build",
    "go build", "npm run build", "npm ci", "npm install", "yarn build",
    "yarn install", "pnpm build", "pnpm install", "tsc", "webpack",
    "vite build", "docker build", "pip install", "poetry install",
    "cargo check", "dotnet build",
)


def classify_command(command: str) -> str | None:
    """'test' | 'build' | None — which host pool this command belongs to."""
    normalized = " ".join((command or "").split()).lower()
    if not normalized:
        return None
    padded = f" {normalized} "
    for marker in HEAVY_TEST_MARKERS:
        if marker in padded:
            return "test"
    for marker in HEAVY_BUILD_MARKERS:
        if marker in padded:
            return "build"
    return None


def tool_schemas_for(role: Role) -> list[dict]:
    """The tool catalog for one role: fixed content, fixed order."""
    if role in _WRITER_ROLES:
        return READ_TOOLS + WRITE_TOOLS
    return list(READ_TOOLS)


def tool_schema_hash(role: Role) -> str:
    """Fingerprint of the serialized catalog — part of the PrefixGroupKey."""
    return sha256_hex(canonical_json(tool_schemas_for(role)))


# -- security decision ------------------------------------------------------

LOCAL_WRITE_TOOLS = {"write_file", "edit_file"}
# Read tools take a path too. Confining writes but not reads would let a
# prompt in an untrusted repository pull ~/.ssh keys or a host .env into
# the model request and the artifacts — the very files the command policy
# blocks on the shell side.
LOCAL_READ_PATH_TOOLS = {"read_file", "list_directory", "grep"}


def resolve_inside(repo_root: str | Path, path: str) -> Path | None:
    """Absolute path for `path`, or None when it escapes `repo_root`."""
    root = Path(repo_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    try:
        resolved = target.resolve()
    except OSError:
        return None
    if resolved != root and root not in resolved.parents:
        return None
    return resolved


def decide_local_tool_use(
    role: Role, tool_name: str, arguments: dict, repo_root: str | Path
) -> tuple[bool, str]:
    """(allowed, reason) for a local tool call. Deny wins over any prompt."""
    if tool_name == "run_command":
        command = arguments.get("command", "")
        decision = check_command(command)
        if not decision.allowed:
            return False, f"Forbidden command: {decision.reason}"
        if role != Role.DEVELOPER:
            hint = find_write_hint(command)
            if hint:
                return False, (
                    f"{role.value} has read-only shell access ({hint}); "
                    "use the write_file/edit_file tools if your role permits file changes"
                )
        return True, ""

    if tool_name in LOCAL_READ_PATH_TOOLS:
        # grep's path is optional (defaults to the repository root).
        file_path = arguments.get("path")
        if file_path and resolve_inside(repo_root, file_path) is None:
            return False, (
                f"Reads outside the repository are forbidden: {file_path}"
            )
        return True, ""

    if tool_name == "glob":
        pattern = arguments.get("pattern", "")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            return False, (
                f"Glob patterns must stay inside the repository: {pattern}"
            )
        return True, ""

    if tool_name in LOCAL_WRITE_TOOLS:
        file_path = arguments.get("path", "")
        if not file_path:
            return False, "path is required"
        repo_root = Path(repo_root).resolve()
        target = Path(file_path)
        if not target.is_absolute():
            target = repo_root / target
        try:
            relative = target.resolve().relative_to(repo_root)
        except ValueError:
            return False, f"Writes outside the repository are forbidden: {file_path}"
        allowed, reason = check_write_path(role, str(relative))
        if not allowed:
            return False, reason
        if str(relative).startswith(".git/") or str(relative) == ".git":
            return False, "Modifying .git internals is forbidden"
        return True, ""

    return True, ""


# -- execution --------------------------------------------------------------

TRUNCATION_NOTICE = "\n... (output truncated)"


@dataclass
class LocalToolExecutor:
    role: Role
    cwd: Path
    repo_root: Path
    guard: RepeatActionGuard | None = None
    command_timeout: int = 300
    max_output_chars: int = MAX_TOOL_OUTPUT_CHARS
    # Shared host pools (asyncio.Semaphore-like). Agent-triggered builds and
    # test runs are gated by the SAME limits as harness verification.
    heavy_build_pool: object | None = None
    heavy_test_pool: object | None = None
    tool_calls: int = 0
    loop_detected: bool = field(default=False)

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """Run one tool call; returns the tool result text (errors included
        as text so the model can react)."""
        self.tool_calls += 1
        allowed, reason = decide_local_tool_use(
            self.role, tool_name, arguments, self.repo_root
        )
        if allowed and self.guard is not None:
            allowed, reason = self.guard.observe(tool_name, arguments)
            if self.guard.loop_detected:
                self.loop_detected = True
        if not allowed:
            return f"TOOL DENIED: {reason}"
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return f"TOOL ERROR: unknown tool {tool_name}"
            result = await handler(arguments)
        except Exception as exc:
            return f"TOOL ERROR: {type(exc).__name__}: {exc}"
        return self._bound(result)

    def _bound(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + TRUNCATION_NOTICE

    def _resolve(self, path: str) -> Path:
        """Resolve inside the repository, or refuse.

        Enforced here as well as in the policy so a future caller cannot
        reach outside by skipping decide_local_tool_use().
        """
        resolved = resolve_inside(self.repo_root, path)
        if resolved is None:
            raise ValueError(f"path escapes the repository: {path}")
        return resolved

    # -- handlers ----------------------------------------------------------

    async def _tool_read_file(self, args: dict) -> str:
        target = self._resolve(args["path"])
        if not target.is_file():
            return f"file not found: {args['path']}"
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        offset = max(int(args.get("offset") or 1), 1)
        limit = int(args.get("limit") or 2000)
        window = lines[offset - 1 : offset - 1 + limit]
        return "\n".join(f"{offset + i}\t{line}" for i, line in enumerate(window))

    async def _tool_list_directory(self, args: dict) -> str:
        target = self._resolve(args["path"])
        if not target.is_dir():
            return f"not a directory: {args['path']}"
        entries = sorted(target.iterdir(), key=lambda p: p.name)
        return "\n".join(
            f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries
        )

    async def _tool_glob(self, args: dict) -> str:
        root = Path(self.repo_root).resolve()
        matches = sorted(
            str(p.relative_to(self.cwd))
            for p in self.cwd.glob(args["pattern"])
            # A pattern can still walk out through a symlink; drop anything
            # that does not resolve inside the repository.
            if ".git" not in p.parts and resolve_inside(root, str(p)) is not None
        )[:500]
        return "\n".join(matches) if matches else "no matches"

    async def _tool_grep(self, args: dict) -> str:
        try:
            regex = re.compile(args["pattern"])
        except re.error as exc:
            return f"invalid pattern: {exc}"
        base = self._resolve(args.get("path") or ".")
        root = Path(self.repo_root).resolve()
        file_glob = args.get("glob")
        results: list[str] = []
        files = [base] if base.is_file() else [
            p for p in sorted(base.rglob("*"))
            if p.is_file() and ".git" not in p.parts
            and resolve_inside(root, str(p)) is not None
        ]
        for path in files:
            rel = str(path.relative_to(self.cwd)) if path.is_relative_to(self.cwd) else str(path)
            if file_glob and not fnmatch.fnmatch(rel, file_glob):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{lineno}:{line.strip()[:300]}")
                    if len(results) >= 500:
                        return "\n".join(results)
        return "\n".join(results) if results else "no matches"

    def _pool_for(self, command: str):
        kind = classify_command(command)
        if kind == "test":
            return self.heavy_test_pool
        if kind == "build":
            return self.heavy_build_pool
        return None

    async def _tool_run_command(self, args: dict) -> str:
        timeout = min(int(args.get("timeout_seconds") or self.command_timeout),
                      self.command_timeout)

        def run() -> str:
            # Own process group, killed as a group on timeout: a test runner's
            # workers must not survive into a worktree the harness is about
            # to archive or delete.
            result = run_command(args["command"], self.cwd, timeout)
            if result.timed_out:
                return result.output
            return f"exit code: {result.exit_code}\n{result.output}"

        # The command runs inside this task's worktree; cancelling the await
        # cannot stop it, so cleanup must not race a live process (see
        # harness.concurrency).
        pool = self._pool_for(args["command"])
        if pool is None:
            return await run_thread_uninterruptible(run, label="agent command")
        async with pool:
            return await run_thread_uninterruptible(
                run, label="agent host-heavy command")

    async def _tool_write_file(self, args: dict) -> str:
        target = self._resolve(args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        return f"wrote {len(args['content'])} chars to {args['path']}"

    async def _tool_edit_file(self, args: dict) -> str:
        target = self._resolve(args["path"])
        if not target.is_file():
            return f"file not found: {args['path']}"
        text = target.read_text(encoding="utf-8")
        old, new = args["old_string"], args["new_string"]
        count = text.count(old)
        if count == 0:
            return "old_string not found in file"
        if count > 1 and not args.get("replace_all"):
            return f"old_string occurs {count} times; make it unique or set replace_all"
        target.write_text(text.replace(old, new), encoding="utf-8")
        return f"replaced {count if args.get('replace_all') else 1} occurrence(s)"
