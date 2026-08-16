"""Node.js verification presets."""

from __future__ import annotations

import json
from pathlib import Path


def default_commands(repo_path: Path) -> list[str]:
    package_json = repo_path / "package.json"
    if not package_json.exists():
        return []
    try:
        scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
    except (json.JSONDecodeError, OSError):
        return []
    commands = []
    if "build" in scripts:
        commands.append("npm run build")
    if "lint" in scripts:
        commands.append("npm run lint")
    if "test" in scripts:
        commands.append("npm test")
    return commands
