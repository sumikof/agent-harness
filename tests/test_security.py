from harness.orchestrator.state_machine import Role
from harness.security.commands import check_command, find_write_hint
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

    def test_git_global_options_do_not_bypass(self):
        # git accepts [-C <path>] [-c <k>=<v>] [--git-dir=...] before the
        # subcommand; the hook must still catch the lifecycle command.
        for cmd in [
            "git -C . commit -m x",
            "git -c user.name=x commit -m x",
            "git -c user.email=a@b push origin main",
            "git --git-dir=.git reset --hard HEAD",
            "git -C /repo -c core.autocrlf=false rebase main",
            "git -C sub branch -D feature",
        ]:
            assert not check_command(cmd).allowed, cmd

    def test_commit_producing_subcommands_forbidden(self):
        # cherry-pick / revert / am create commits; update-ref & friends move
        # harness-owned ref state directly.
        for cmd in [
            "git cherry-pick abc123",
            "git revert HEAD",
            "git am patch.mbox",
            "git update-ref refs/heads/main abc123",
            "git symbolic-ref HEAD refs/heads/other",
            "git filter-branch --all",
            "git worktree add ../scratch",
            "git branch -f main HEAD~3",
            "git branch -m old new",
            "git reflog expire --all",
            "git gc --prune=now",
        ]:
            assert not check_command(cmd).allowed, cmd
        # plain listing stays allowed
        assert check_command("git branch").allowed
        assert check_command("git branch --list").allowed

    def test_branch_copy_forbidden(self):
        assert not check_command("git branch -c main copy").allowed
        assert not check_command("git branch -C main copy").allowed
        assert not check_command("git branch --copy main copy").allowed

    def test_branch_clustered_flags_forbidden(self):
        # mutation letters hidden inside short-option clusters
        for cmd in [
            "git branch -fc old copied",
            "git branch -df copied",
            "git branch -vd feature",
        ]:
            assert not check_command(cmd).allowed, cmd
        # harmless listing options stay allowed
        for cmd in ["git branch -v", "git branch -a", "git branch -r", "git branch --merged"]:
            assert check_command(cmd).allowed, cmd

    def test_optionless_branch_creation_forbidden(self):
        for cmd in [
            "git branch new-feature",
            "git branch new-feature HEAD~2",
            "git -C repo branch topic",
        ]:
            assert not check_command(cmd).allowed, cmd
        # flags-only listing still allowed
        assert check_command("git branch").allowed
        assert check_command("git branch -av").allowed

    def test_shell_escaping_does_not_bypass(self):
        # the shell resolves these to plain `git commit` etc. before executing
        for cmd in [
            r"g\it commit -m x",
            r"git co\mmit -m x",
            "'git' commit -m x",
            '"git" "push" origin main',
            r"g\it -C . push origin main",
        ]:
            assert not check_command(cmd).allowed, cmd
        # unparseable quoting fails closed
        assert not check_command("git 'unclosed quote").allowed

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


class TestWriteHints:
    def test_write_commands_detected(self):
        for cmd in [
            "echo x > src/app.py",
            "cat a.txt >> notes.md",
            "printf 'x' > file",
            "grep foo src | tee out.txt",
            "sed -i 's/a/b/' src/x.py",
            "perl -i -pe 's/a/b/' src/x.py",
            "mv a.py b.py",
            "cp src/a.py src/b.py",
            "rm src/a.py",
            "touch marker",
            "mkdir newdir",
            "pip install requests",
            # stderr / fd-numbered redirects to regular files are writes too
            "pytest 2> src/main.py",
            "make 2>> build-errors.log",
            "cmd &> capture.txt",
            "pytest >& src/main.py",
            r"s\ed -i 's/a/b/' src/x.py",
        ]:
            assert find_write_hint(cmd) is not None, cmd

    def test_read_only_commands_not_flagged(self):
        for cmd in [
            "cat src/app.py",
            "grep -r 'TODO' src",
            "python -m pytest -q 2>/dev/null",
            "ls -la > /dev/null",
            "./mvnw test 2>&1 | tail -20",
            "git status",
            "sed -n '1,10p' src/x.py",
            "make 2> /dev/null",
            "cmd &> /dev/null",
            "grep -- '->' src/x.py",
            "cmd >& /dev/null",
            "cmd >&2",
        ]:
            assert find_write_hint(cmd) is None, cmd


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

    def test_read_only_role_bash_write_denied(self, tmp_path):
        for role in (Role.PLANNER, Role.ANALYST, Role.REVIEWER, Role.DIAGNOSTICIAN, Role.TESTER):
            allowed, reason = decide_tool_use(
                role, "Bash", {"command": "echo x > src/app.py"}, tmp_path
            )
            assert not allowed and "read-only" in reason, role

    def test_read_only_role_bash_read_allowed(self, tmp_path):
        for role in (Role.PLANNER, Role.ANALYST, Role.REVIEWER, Role.DIAGNOSTICIAN, Role.TESTER):
            allowed, _ = decide_tool_use(
                role, "Bash", {"command": "python -m pytest -q"}, tmp_path
            )
            assert allowed, role

    def test_developer_bash_write_allowed(self, tmp_path):
        allowed, _ = decide_tool_use(
            Role.DEVELOPER, "Bash", {"command": "echo x > src/app.py"}, tmp_path
        )
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
