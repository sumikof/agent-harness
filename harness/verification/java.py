"""Java verification presets."""

from __future__ import annotations

from pathlib import Path


def default_commands(repo_path: Path) -> list[str]:
    mvnw = repo_path / "mvnw"
    if mvnw.exists():
        return ["./mvnw -B compile", "./mvnw -B test"]
    if (repo_path / "gradlew").exists():
        return ["./gradlew build -x test", "./gradlew test"]
    if (repo_path / "pom.xml").exists():
        return ["mvn -B compile", "mvn -B test"]
    return []
