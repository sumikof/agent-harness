"""CLI entry point.

    agent-harness init   --config config.yaml   # prepare workspace
    agent-harness run    --config config.yaml   # run / resume the project loop
    agent-harness status --config config.yaml   # show project & task state
    agent-harness events --config config.yaml   # show recent event log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import HarnessConfig, load_config
from .database.connection import Database
from .database.event_repository import EventRepository
from .database.project_repository import ProjectRepository
from .database.task_repository import TaskRepository
from .git.repository import GitRepository
from .orchestrator.project import ProjectOrchestrator
from .orchestrator.recovery import UnexplainedDirtyWorktree
from .workspace_lock import UnsupportedPlatform, WorkspaceLocked


def setup_logging(config: HarnessConfig) -> None:
    config.logs_path.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.logs_path / "harness.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def cmd_init(config: HarnessConfig) -> int:
    config.workspace_path.mkdir(parents=True, exist_ok=True)
    for sub in ("artifacts", "logs"):
        (config.workspace_path / sub).mkdir(exist_ok=True)
    Database(config.db_path).close()

    repo = GitRepository(config.repository_path)
    if not repo.is_repo():
        print(f"NOTE: {config.repository_path} is not a git repository yet.")
        print("      Clone or copy the target repository there before `run`.")
    print(f"workspace initialized at {config.workspace_path}")
    return 0


def _uses_local_inference(config: HarnessConfig) -> bool:
    local_types = {"openai-compatible", "openai_compatible", "vllm"}
    if config.provider.type in local_types:
        return True
    return any((role.type or config.provider.type) in local_types
               for role in config.provider.roles.values())


def cmd_health(config: HarnessConfig) -> int:
    """Verify the local inference endpoint (liveness, model, completion,
    tool calling, structured output, prefix cache) before any task runs."""
    from .agents.health import verify_endpoint

    report = asyncio.run(verify_endpoint(config.inference))
    print(f"endpoint          : {config.inference.base_url}")
    print(f"model available   : {report.model_available}")
    print(f"completion        : {report.completion_ok} (usage={report.usage_reported})")
    print(f"tool calling      : {report.tool_calling_ok}")
    print(f"structured output : {report.structured_output_ok}")
    print(f"reasoning content : {report.reasoning_content_seen}")
    print(f"prefix cache      : {'verified' if report.prefix_cache_verified else report.prefix_cache_note or 'unverified'}")
    for error in report.errors:
        print(f"ERROR: {error}")
    return 0 if report.ok else 1


def cmd_run(config: HarnessConfig) -> int:
    # A misconfigured/absent inference server must not start long-running
    # tasks: the endpoint is verified (real smoke tests, not just a ping)
    # before the orchestrator touches any state.
    if _uses_local_inference(config):
        if cmd_health(config) != 0:
            print("refusing to start: inference endpoint failed verification "
                  "(see deploy/inference/healthcheck.sh)")
            return 1
    try:
        orchestrator = ProjectOrchestrator(config)
        final_state = asyncio.run(orchestrator.run())
    except (UnexplainedDirtyWorktree, WorkspaceLocked, UnsupportedPlatform) as exc:
        print(f"refusing to start: {exc}")
        return 1
    print(f"project finished in state: {final_state}")
    return 0 if final_state.value in ("COMPLETED", "PAUSED") else 1


def cmd_status(config: HarnessConfig) -> int:
    if not config.db_path.exists():
        print("no harness.db yet — run `agent-harness init` first")
        return 1
    db = Database(config.db_path)
    projects = ProjectRepository(db)
    tasks = TaskRepository(db)
    project = projects.get_by_name(config.project.name)
    if project is None:
        print(f"project '{config.project.name}' not found")
        return 1
    print(f"project : {project['name']}")
    print(f"status  : {project['status']}")
    print(f"budget  : ${project['spent_usd']:.2f} / ${project['budget_usd']:.2f}")
    print()
    print(f"{'seq':>5}  {'key':<8} {'status':<16} {'att':>3}  {'commit':<10} title")
    for task in tasks.list_for_project(project["id"]):
        commit = (task["current_commit"] or "")[:8]
        print(
            f"{task['sequence']:>5}  {task['task_key']:<8} {task['status']:<16} "
            f"{task['attempt_count']:>3}  {commit:<10} {task['title']}"
        )
    db.close()
    return 0


def cmd_events(config: HarnessConfig, limit: int) -> int:
    if not config.db_path.exists():
        print("no harness.db yet")
        return 1
    db = Database(config.db_path)
    projects = ProjectRepository(db)
    events = EventRepository(db)
    project = projects.get_by_name(config.project.name)
    if project is None:
        print(f"project '{config.project.name}' not found")
        return 1
    rows = events.list_for_project(project["id"], limit=limit)
    for row in reversed(rows):
        payload = json.loads(row["payload"] or "{}")
        extra = f" {payload}" if payload else ""
        print(f"{row['created_at']}  {row['event_type']}{extra}")
    db.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-harness",
                                     description="Long-running coding agent harness")
    parser.add_argument("--config", "-c", default="config.yaml", help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="prepare the workspace")
    sub.add_parser("run", help="run or resume the project loop")
    sub.add_parser("health", help="verify the local inference endpoint")
    sub.add_parser("status", help="show project and task state")
    events_parser = sub.add_parser("events", help="show recent events")
    events_parser.add_argument("--limit", type=int, default=50)

    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config not found: {config_path} (copy config.example.yaml to get started)")
        return 1
    config = load_config(config_path)
    setup_logging(config)

    if args.command == "init":
        return cmd_init(config)
    if args.command == "run":
        return cmd_run(config)
    if args.command == "health":
        return cmd_health(config)
    if args.command == "status":
        return cmd_status(config)
    if args.command == "events":
        return cmd_events(config, args.limit)
    return 1


if __name__ == "__main__":
    sys.exit(main())
