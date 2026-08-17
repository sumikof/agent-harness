from pathlib import Path

from harness.config import HarnessConfig, ProjectConfig


def test_prompts_fall_back_to_packaged_files(tmp_path):
    """Installed wheels have no source checkout; prompts ship in the package."""
    cfg = HarnessConfig(project=ProjectConfig(name="x"), workspace_dir=str(tmp_path))
    cfg.config_path = tmp_path / "config.yaml"  # no prompts/ next to it
    prompts = cfg.prompts_path
    for name in ("planner", "analyst", "developer", "tester", "reviewer", "diagnostician"):
        assert (prompts / f"{name}.md").is_file(), name


def test_local_prompts_directory_overrides_package(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "planner.md").write_text("custom", encoding="utf-8")
    cfg = HarnessConfig(project=ProjectConfig(name="x"), workspace_dir=str(tmp_path))
    cfg.config_path = tmp_path / "config.yaml"
    assert cfg.prompts_path == tmp_path / "prompts"
