"""Python verification presets."""

from __future__ import annotations

from pathlib import Path


def default_commands(repo_path: Path) -> list[str]:
    commands: list[str] = []
    if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
        commands.append("python -m compileall -q .")
    if (
        (repo_path / "pytest.ini").exists()
        or (repo_path / "tests").is_dir()
        or (repo_path / "pyproject.toml").exists()
    ):
        commands.append("python -m pytest -q")
    return commands or ["python -m compileall -q ."]
