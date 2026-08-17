from harness.orchestrator.state_machine import Role
from harness.security.commands import check_command
from harness.security.hooks import decide_tool_use
from harness.security.permissions import allowed_tools_for, check_write_path, is_test_path


class TestCommandPolicy:
    def test_git_lifecycle_forbidden(self):
        for cmd in [
            "git push origin main",
            "git commit -m 'x'",
            "git merge feature",
            "git rebase main",
            "git reset --hard HEAD",
            "git branch -D feature",
            "git remote add origin x",
        ]:
            assert not check_command(cmd).allowed, cmd

    def test_dangerous_commands_forbidden(self):
        for cmd in [
            "sudo apt install x",
            "rm -rf /",
            "cat ~/.ssh/id_rsa",
            "cat .env",
            "docker system prune -af",
            "echo $ANTHROPIC_API_KEY",
        ]:
            assert not check_command(cmd).allowed, cmd

    def test_normal_commands_allowed(self):
        for cmd in [
            "git status",
            "git diff",
            "git log --oneline",
            "ls -la src/",
            "python -m pytest -q",
            "./mvnw test",
            "npm test",
            "grep -r 'TODO' src",
            "rm build/output.txt",
        ]:
            assert check_command(cmd).allowed, cmd


class TestPermissions:
    def test_read_only_roles_have_no_write_tools(self):
        for role in (Role.PLANNER, Role.ANALYST, Role.REVIEWER, Role.DIAGNOSTICIAN):
            tools = allowed_tools_for(role)
            assert "Edit" not in tools and "Write" not in tools

    def test_developer_can_write(self):
        assert "Edit" in allowed_tools_for(Role.DEVELOPER)

    def test_test_paths(self):
        assert is_test_path("tests/test_foo.py")
        assert is_test_path("src/test/java/FooTest.java")
        assert is_test_path("pkg/__tests__/foo.test.ts")
        assert is_test_path("app/module/tests/test_bar.py")
        assert not is_test_path("src/main/java/Foo.java")
        assert not is_test_path("harness/main.py")

    def test_tester_write_restrictions(self):
        ok, _ = check_write_path(Role.TESTER, "tests/test_x.py")
        assert ok
        ok, reason = check_write_path(Role.TESTER, "src/main.py")
        assert not ok and "test" in reason.lower()

    def test_developer_write_anywhere_in_repo(self):
        ok, _ = check_write_path(Role.DEVELOPER, "src/main.py")
        assert ok


class TestToolUseDecision:
    def test_bash_forbidden_command_denied(self, tmp_path):
        allowed, reason = decide_tool_use(
            Role.DEVELOPER, "Bash", {"command": "git commit -m hi"}, tmp_path
        )
        assert not allowed and "harness" in reason

    def test_bash_normal_allowed(self, tmp_path):
        allowed, _ = decide_tool_use(Role.DEVELOPER, "Bash", {"command": "pytest -q"}, tmp_path)
        assert allowed

    def test_write_outside_repo_denied(self, tmp_path):
        allowed, _ = decide_tool_use(
            Role.DEVELOPER, "Write", {"file_path": "/etc/passwd"}, tmp_path
        )
        assert not allowed

    def test_tester_write_prod_denied(self, tmp_path):
        allowed, _ = decide_tool_use(
            Role.TESTER, "Edit", {"file_path": str(tmp_path / "src" / "app.py")}, tmp_path
        )
        assert not allowed

    def test_tester_write_test_allowed(self, tmp_path):
        allowed, _ = decide_tool_use(
            Role.TESTER, "Edit", {"file_path": str(tmp_path / "tests" / "test_app.py")}, tmp_path
        )
        assert allowed

    def test_git_internals_denied(self, tmp_path):
        allowed, _ = decide_tool_use(
            Role.DEVELOPER, "Write", {"file_path": str(tmp_path / ".git" / "config")}, tmp_path
        )
        assert not allowed
