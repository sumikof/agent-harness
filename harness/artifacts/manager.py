"""Filesystem artifact store.

SQLite holds state, indexes, and small JSON; long logs and agent outputs
live on the filesystem under workspace/artifacts/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from .schemas import ArtifactEnvelope, ArtifactProducer, SpilledOutput

T = TypeVar("T", bound=BaseModel)

# Above this many characters an output is spilled to a file and only a
# bounded preview travels in agent context. Callers may override per site.
DEFAULT_SPILL_THRESHOLD = 30000
SPILL_HEAD_CHARS = 8000
SPILL_TAIL_CHARS = 8000


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ArtifactManager:
    def __init__(self, artifacts_dir: str | Path):
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "tasks").mkdir(exist_ok=True)
        (self.root / "diagnostics").mkdir(exist_ok=True)
        (self.root / "manifests").mkdir(exist_ok=True)
        (self.root / "spill").mkdir(exist_ok=True)

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
        data = self.unwrap_envelope(data)
        return model_type.model_validate(data)

    # -- provenance envelope ------------------------------------------------

    def save_enveloped(
        self,
        path: Path,
        model: BaseModel,
        *,
        artifact_type: Optional[str] = None,
        project_id: Optional[int] = None,
        task_id: Optional[int] = None,
        attempt_no: Optional[int] = None,
        producer_role: Optional[str] = None,
        producer_run_id: Optional[int] = None,
        base_commit: Optional[str] = None,
        input_manifest_hash: Optional[str] = None,
        created_at: str = "",
    ) -> Path:
        envelope = ArtifactEnvelope(
            artifact_type=artifact_type or type(model).__name__,
            project_id=project_id,
            task_id=task_id,
            attempt_no=attempt_no,
            producer=ArtifactProducer(role=producer_role, agent_run_id=producer_run_id)
            if producer_role
            else None,
            base_commit=base_commit,
            input_manifest_hash=input_manifest_hash,
            created_at=created_at,
            payload=model.model_dump(mode="json"),
        )
        return self.save_json(path, envelope.model_dump(mode="json"))

    @staticmethod
    def unwrap_envelope(data: dict | list) -> dict | list:
        """Accept both enveloped and legacy raw artifact documents."""
        if isinstance(data, dict) and "artifact_type" in data and "payload" in data:
            return data["payload"]
        return data

    # -- large output retention ---------------------------------------------

    def spill_text_output(
        self,
        name: str,
        text: str,
        *,
        threshold: int = DEFAULT_SPILL_THRESHOLD,
        head_chars: int = SPILL_HEAD_CHARS,
        tail_chars: int = SPILL_TAIL_CHARS,
    ) -> SpilledOutput:
        """Persist `text` fully under artifacts/spill/ and return a bounded
        preview safe to place in agent context. The full output is never lost.

        The head+tail preview is capped at `threshold` characters total, so a
        deployment that lowers the inline limit gets a preview that actually
        honors it — never more context than an un-spilled output would use.
        """
        path = self.root / "spill" / name
        self.save_text(path, text)
        truncated = len(text) > threshold
        head_chars = min(head_chars, max(1, threshold // 2))
        tail_chars = min(tail_chars, max(0, threshold - head_chars))
        # text[-0:] is the WHOLE string, not "" — guard the zero-tail case.
        tail = "" if not truncated or tail_chars <= 0 else text[-tail_chars:]
        return SpilledOutput(
            artifact_path=str(path),
            sha256=sha256_text(text),
            total_bytes=len(text.encode("utf-8")),
            truncated=truncated,
            head=text if not truncated else text[:head_chars],
            tail=tail,
        )

    def save_text(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def save_bytes(self, path: Path, data: bytes) -> Path:
        """For content that is not guaranteed UTF-8 (e.g. recovery patches)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def archive_worktree_files(
        self, tar_path: Path, repo_root: Path, paths: list[str]
    ) -> Optional[Path]:
        """Tar the actual on-disk files (byte-exact, no git filters).

        A git patch passes through the repository's clean filters (LFS,
        redaction filters, ...), so it may not reproduce the working-tree
        bytes. The tar is the ground-truth copy alongside the patch.
        Returns the tar path, or None when nothing existed to archive.

        Nodes that cannot be captured faithfully (sockets, devices) raise
        instead of being silently omitted — the caller must NOT reset a
        tree it could not fully archive.
        """
        import os
        import stat
        import tarfile

        def entries(rel: str):
            full = repo_root / rel
            if full.is_dir() and not full.is_symlink():
                for walk_root, walk_dirs, names in os.walk(full):
                    for name in names:
                        yield str((Path(walk_root) / name).relative_to(repo_root))
                    # symlinks to directories show up in dirs (never
                    # descended) — they are entries of their own
                    for name in walk_dirs:
                        candidate = Path(walk_root) / name
                        if candidate.is_symlink():
                            yield str(candidate.relative_to(repo_root))
            else:
                yield rel

        flat: list[str] = []
        for rel in paths:
            if not (repo_root / rel).exists() and not (repo_root / rel).is_symlink():
                continue
            for entry in entries(rel):
                full = repo_root / entry
                try:
                    node = os.lstat(full)
                except FileNotFoundError:
                    continue
                mode = node.st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode) or stat.S_ISFIFO(mode)):
                    raise ValueError(
                        f"cannot archive special node at {entry} (socket/device); "
                        "refusing to proceed without a faithful copy"
                    )
                flat.append(entry)
        if not flat:
            return None
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "w") as tar:
            for entry in sorted(set(flat)):
                tar.add(repo_root / entry, arcname=entry, recursive=False)
        return tar_path

    def relpath(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def resolve(self, stored_path: str | Path) -> Path:
        """Resolve a stored artifact reference against the current root.

        References are stored workspace-relative so a moved or restored
        workspace keeps working; absolute paths (legacy records) pass
        through unchanged.
        """
        path = Path(stored_path)
        return path if path.is_absolute() else self.root / path
