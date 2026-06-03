"""Central configuration. No secrets here — the API key is read from the
environment so it never enters the model's context or the repo."""
import os
from pathlib import Path

# --- Models -----------------------------------------------------------------
CORE_MODEL = os.environ.get("DP_CORE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("DP_MAX_TOKENS", "2048"))

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MEMORY_DIR = ROOT / "memory"
WORKSPACE_DIR = ROOT / "workspace"        # the ONLY dir file tools may touch
VECTOR_DIR = MEMORY_DIR / "vectorstore"   # Chroma persistence (derived index)
AUDIT_LOG = MEMORY_DIR / "audit.log"

EPISODIC_LOG = MEMORY_DIR / "episodic.jsonl"
FACTS_FILE = MEMORY_DIR / "facts.json"        # canonical structured facts
SEMANTIC_MD = MEMORY_DIR / "semantic.md"      # human-readable, inspectable view
AFFECT_FILE = MEMORY_DIR / "affect.json"
SKILLS_FILE = MEMORY_DIR / "skills.json"      # procedural memory

# --- Memory tuning ----------------------------------------------------------
RECENT_EPISODES = 8
TOP_FACTS = 6
DECAY_PER_SLEEP = 0.9

# --- Sandbox ----------------------------------------------------------------
# Where tool/command execution runs: "local" | "docker" | "ssh".
SANDBOX_BACKEND = os.environ.get("DP_SANDBOX", "local")
CMD_TIMEOUT = int(os.environ.get("DP_CMD_TIMEOUT", "30"))
DOCKER_IMAGE = os.environ.get("DP_DOCKER_IMAGE", "python:3.12-slim")
SSH_TARGET = os.environ.get("DP_SSH_TARGET", "")          # e.g. user@host
SSH_REMOTE_DIR = os.environ.get("DP_SSH_REMOTE_DIR", "~/dp-workspace")

for d in (MEMORY_DIR, WORKSPACE_DIR, VECTOR_DIR):
    d.mkdir(parents=True, exist_ok=True)
