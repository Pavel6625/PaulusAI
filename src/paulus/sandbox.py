"""Pluggable execution backends — inspired by Hermes Agent's multi-backend
sandboxing (local, Docker, SSH, ...).

The key security idea: file and command execution go through a Sandbox, not
straight to the host. The local backend confines paths to the workspace; the
Docker backend additionally isolates command execution in a network-disabled
container with the workspace mounted; the SSH backend runs on a remote host.

File reads/writes operate on the shared workspace directory in every backend
(for Docker it's the mounted volume); only command *execution* changes per
backend, which is where isolation matters most. Select via config.SANDBOX_BACKEND.
"""
import shlex
import subprocess

from . import config


class Sandbox:
    def safe_path(self, rel):
        config.ensure_dirs()
        target = (config.WORKSPACE_DIR / rel).resolve()
        root = str(config.WORKSPACE_DIR.resolve())
        if not str(target).startswith(root):
            raise ValueError("path escapes the sandboxed workspace")
        return target

    def read_file(self, rel):
        return self.safe_path(rel).read_text(encoding="utf-8")

    def write_file(self, rel, content):
        p = self.safe_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {rel}."

    def run(self, command):
        raise NotImplementedError


class LocalSandbox(Sandbox):
    """Runs commands as a subprocess in the workspace dir. No isolation beyond
    the working directory and a timeout — fine for trusted local dev, but use
    Docker/SSH for anything that touches untrusted input."""

    def run(self, command):
        config.ensure_dirs()
        proc = subprocess.run(
            command, shell=True, cwd=str(config.WORKSPACE_DIR),
            capture_output=True, text=True, timeout=config.CMD_TIMEOUT,
        )
        return (proc.stdout + proc.stderr).strip() or "(no output)"


class DockerSandbox(Sandbox):
    """Runs commands inside a throwaway, network-disabled container with the
    workspace mounted at /work. Requires Docker installed and config.DOCKER_IMAGE."""

    def run(self, command):
        config.ensure_dirs()
        docker_cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--cpus", "1", "--memory", "512m",
            "-v", f"{config.WORKSPACE_DIR.resolve()}:/work", "-w", "/work",
            config.DOCKER_IMAGE, "sh", "-c", command,
        ]
        proc = subprocess.run(docker_cmd, capture_output=True, text=True,
                              timeout=config.CMD_TIMEOUT)
        return (proc.stdout + proc.stderr).strip() or "(no output)"


class SSHSandbox(Sandbox):
    """Runs commands on a remote host over SSH (set config.SSH_TARGET and
    config.SSH_REMOTE_DIR). File ops still use the local workspace; sync is left
    to the operator (e.g. rsync) — kept minimal for the MVP."""

    def run(self, command):
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
