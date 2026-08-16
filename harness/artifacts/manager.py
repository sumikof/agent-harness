"""Filesystem artifact store.

SQLite holds state, indexes, and small JSON; long logs and agent outputs
live on the filesystem under workspace/artifacts/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ArtifactManager:
    def __init__(self, artifacts_dir: str | Path):
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "tasks").mkdir(exist_ok=True)
        (self.root / "diagnostics").mkdir(exist_ok=True)

    # -- path helpers ------------------------------------------------------

    def task_dir(self, task_key: str) -> Path:
        path = self.root / "tasks" / task_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_artifact_path(self, task_key: str, name: str) -> Path:
        return self.task_dir(task_key) / name

    def project_plan_path(self) -> Path:
        return self.root / "project-plan.json"

    def diagnostics_path(self, task_key: str, attempt_no: int) -> Path:
        return self.root / "diagnostics" / f"{task_key}-attempt{attempt_no}.json"

    # -- json io -----------------------------------------------------------

    def save_json(self, path: Path, data: dict | list) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_json(self, path: Path) -> Optional[dict | list]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_model(self, path: Path, model: BaseModel) -> Path:
        return self.save_json(path, model.model_dump(mode="json"))

    def load_model(self, path: Path, model_type: Type[T]) -> Optional[T]:
        data = self.load_json(path)
        if data is None:
            return None
        return model_type.model_validate(data)

    def save_text(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def relpath(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)
