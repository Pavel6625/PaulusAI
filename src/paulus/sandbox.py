"""Pluggable execution backends — inspired by Hermes Agent's multi-backend
sandboxing (local, Docker, SSH, ...).

The key security idea: file and command execution go through a Sandbox, not
straight to the host. The local backend confines paths to the workspace; the
Docker backend additionally isolates command execution in a network-disabled
container with the workspace mounted; the SSH backend runs on a remote host.

File reads/writes operate on a per-user workspace directory in every backend
(for Docker it's the mounted volume); only command *execution* changes per
backend, which is where isolation matters most. Select via config.SANDBOX_BACKEND.

User isolation: every method takes a *user_id*. Each user gets their own
workspace under ``WORKSPACE_DIR/users/<safe_uid>/`` (see config.user_workspace),
so one user can never read, write, or run against another user's files.
``user_id=None`` (CLI / single-user mode) falls back to the shared
WORKSPACE_DIR, keeping existing single-user installs unaffected. The SSH
backend's *remote* dir is intentionally left global for now (sync is the
operator's responsibility); its local file ops are still per-user.
"""
import shlex
import subprocess

from . import config


class Sandbox:
    def _root(self, user_id):
        root = config.user_workspace(user_id)
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def safe_path(self, rel, user_id=None):
        config.ensure_dirs()
        root = self._root(user_id)
        target = (root / rel).resolve()
        # Confine to the *user's* workspace, not just the shared root, so a
        # crafted rel path can't reach a sibling user's directory.
        if target != root and root not in target.parents:
            raise ValueError("path escapes the sandboxed workspace")
        return target

    def read_file(self, rel, user_id=None):
        return self.safe_path(rel, user_id).read_text(encoding="utf-8")

    def write_file(self, rel, content, user_id=None):
        p = self.safe_path(rel, user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {rel}."

    def run(self, command, user_id=None):
        raise NotImplementedError


class LocalSandbox(Sandbox):
    """Runs commands as a subprocess in the user's workspace dir. No isolation
    beyond the working directory and a timeout — fine for trusted local dev, but
    use Docker/SSH for anything that touches untrusted input."""

    def run(self, command, user_id=None):
        config.ensure_dirs()
        proc = subprocess.run(
            command, shell=True, cwd=str(self._root(user_id)),
            capture_output=True, text=True, timeout=config.CMD_TIMEOUT,
        )
        return (proc.stdout + proc.stderr).strip() or "(no output)"


class DockerSandbox(Sandbox):
    """Runs commands inside a throwaway, network-disabled container with the
    user's workspace mounted at /work. Requires Docker installed and
    config.DOCKER_IMAGE."""

    def run(self, command, user_id=None):
        config.ensure_dirs()
        docker_cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--cpus", "1", "--memory", "512m",
            "-v", f"{self._root(user_id)}:/work", "-w", "/work",
            config.DOCKER_IMAGE, "sh", "-c", command,
        ]
        proc = subprocess.run(docker_cmd, capture_output=True, text=True,
                              timeout=config.CMD_TIMEOUT)
        return (proc.stdout + proc.stderr).strip() or "(no output)"


class SSHSandbox(Sandbox):
    """Runs commands on a remote host over SSH (set config.SSH_TARGET and
    config.SSH_REMOTE_DIR). The remote dir is shared across users for now; sync
    is left to the operator (e.g. rsync) — kept minimal for the MVP. Local file
    ops (read/write) are per-user via the inherited Sandbox methods."""

    def run(self, command, user_id=None):
        remote = f"cd {shlex.quote(config.SSH_REMOTE_DIR)} && {command}"
        proc = subprocess.run(
            ["ssh", config.SSH_TARGET, remote],
            capture_output=True, text=True, timeout=config.CMD_TIMEOUT,
        )
        return (proc.stdout + proc.stderr).strip() or "(no output)"


_BACKENDS = {"local": LocalSandbox, "docker": DockerSandbox, "ssh": SSHSandbox}


def get_sandbox():
    backend = _BACKENDS.get(config.SANDBOX_BACKEND, LocalSandbox)
    return backend()
