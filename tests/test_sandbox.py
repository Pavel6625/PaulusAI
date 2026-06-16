import pytest

from paulus import sandbox


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
