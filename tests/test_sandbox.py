import pytest

from paulus import config, sandbox


def test_local_sandbox_selected_by_default():
    assert isinstance(sandbox.get_sandbox(), sandbox.LocalSandbox)


def test_write_then_read_roundtrip():
    sb = sandbox.LocalSandbox()
    sb.write_file("note.txt", "hello world")
    assert sb.read_file("note.txt") == "hello world"


def test_path_escape_is_blocked():
    sb = sandbox.LocalSandbox()
    with pytest.raises(ValueError):
        sb.safe_path("../../etc/passwd")


def test_run_command_executes():
    out = sandbox.LocalSandbox().run("echo paulus_ok")
    assert "paulus_ok" in out


# --- Per-user isolation -----------------------------------------------------

def test_each_user_gets_a_separate_workspace():
    sb = sandbox.LocalSandbox()
    sb.write_file("note.txt", "alice's note", user_id="alice")
    sb.write_file("note.txt", "bob's note", user_id="bob")
    assert sb.read_file("note.txt", user_id="alice") == "alice's note"
    assert sb.read_file("note.txt", user_id="bob") == "bob's note"


def test_one_user_cannot_read_anothers_file():
    sb = sandbox.LocalSandbox()
    sb.write_file("secret.txt", "alice only", user_id="alice")
    # Bob's workspace has no such file -> not silently served alice's copy.
    with pytest.raises(FileNotFoundError):
        sb.read_file("secret.txt", user_id="bob")


def test_user_cannot_escape_into_a_sibling_user_dir():
    sb = sandbox.LocalSandbox()
    with pytest.raises(ValueError):
        sb.safe_path("../alice/secret.txt", user_id="bob")


def test_malicious_user_id_is_sanitised():
    # A traversal-laden id must not break out of users/ into a sibling.
    bob = sb_root("bob")
    evil = sb_root("../alice")
    assert bob.parent == evil.parent  # both confined under users/


def test_user_workspace_falls_back_to_shared_dir_for_none():
    assert config.user_workspace(None) == config.WORKSPACE_DIR


def test_run_command_is_scoped_to_user_workspace():
    # A command's cwd is the issuing user's workspace: a file it creates lands
    # there and is invisible to another user. `echo > file` works on both
    # cmd.exe and POSIX sh, keeping the test cross-platform.
    sb = sandbox.LocalSandbox()
    sb.run("echo scoped> from_cmd.txt", user_id="alice")
    assert "scoped" in sb.read_file("from_cmd.txt", user_id="alice")
    with pytest.raises(FileNotFoundError):
        sb.read_file("from_cmd.txt", user_id="bob")


def sb_root(user_id):
    return config.user_workspace(user_id).resolve()
