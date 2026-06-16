"""Shared test fixtures.

Crucially, this points all runtime state at a throwaway directory *before*
paulus.config is imported, so tests never touch the real ~/.local/share/paulus.
"""
import os
import shutil
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="paulus-test-")
os.environ["DP_DATA_DIR"] = _TMP
os.environ.setdefault("DP_UNATTENDED_POLICY", "deny")
os.environ.setdefault("DP_CORE_MODEL", "anthropic/claude-sonnet-4-6")
os.environ.setdefault("DP_SANDBOX", "local")


@pytest.fixture(autouse=True)
def clean_data_dir():
    """Give every test empty memory/workspace dirs for isolation."""
    from paulus import config
    for d in (config.MEMORY_DIR, config.WORKSPACE_DIR):
        if d.exists():
            shutil.rmtree(d)
    config.ensure_dirs()
    yield


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
