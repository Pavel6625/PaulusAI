"""Central configuration.

No secrets live here — API keys are read from the environment (or a `.env`
file) so they never enter the model's context or the repository.

Runtime state (memory, audit log, vector index, sandbox workspace) is written
to a *data directory* that is deliberately kept OUTSIDE the installed package,
so an installed copy in site-packages never tries to write to itself. The
location is resolved in this order:

  1. ``$DP_DATA_DIR``                         (explicit override)
  2. ``$XDG_DATA_HOME/paulus``                (Linux/XDG convention)
  3. ``~/.local/share/paulus``                (fallback)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load a .env if present. Priority: explicit DP_ENV_FILE, then the nearest .env
# found by walking up from the current working directory (dev convenience).
_env_file = os.environ.get("DP_ENV_FILE")
if _env_file:
    load_dotenv(_env_file)
else:
    load_dotenv()  # find_dotenv() searches cwd and parents; no-op if absent


def _resolve_data_dir() -> Path:
    explicit = os.environ.get("DP_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "paulus"


# --- Models -----------------------------------------------------------------
# Use LiteLLM provider-prefixed model strings, e.g.:
#   anthropic/claude-sonnet-4-6  (needs ANTHROPIC_API_KEY)
#   openai/gpt-4o                (needs OPENAI_API_KEY)
#   gemini/gemini-1.5-pro        (needs GEMINI_API_KEY)
#   ollama_chat/llama3           (needs Ollama running locally, no key)
CORE_MODEL = os.environ.get("DP_CORE_MODEL", "anthropic/claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("DP_MAX_TOKENS", "2048"))

# --- Paths ------------------------------------------------------------------
DATA_DIR = _resolve_data_dir()
MEMORY_DIR = DATA_DIR / "memory"
WORKSPACE_DIR = DATA_DIR / "workspace"     # the ONLY dir file tools may touch
VECTOR_DIR = MEMORY_DIR / "vectorstore"    # Chroma persistence (derived index)
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
# Facts whose salience decays below this are forgotten (dropped from facts.json
# and the vector index) on the next consolidation. Set to 0 to never evict.
SALIENCE_FLOOR = float(os.environ.get("DP_SALIENCE_FLOOR", "0.05"))
# Hard cap on the episodic log so it can't grow without bound. The log is
# trimmed to the most recent MAX_EPISODES entries; set to 0 to disable.
MAX_EPISODES = int(os.environ.get("DP_MAX_EPISODES", "2000"))

# --- Provider overrides (optional) -----------------------------------------
# Set these when using a non-default API endpoint (e.g. Ollama Cloud).
# DP_API_KEY falls back to OLLAMA_CLOUD_API_KEY so the .env needs no change.
API_BASE = os.environ.get("DP_API_BASE")
API_KEY = (os.environ.get("DP_API_KEY")
           or os.environ.get("OLLAMA_CLOUD_API_KEY"))

# --- Idle / proactive behaviour ---------------------------------------------
IDLE_CHECK_INTERVAL = int(os.environ.get("DP_IDLE_CHECK", "300"))
MIN_IDLE_MINUTES = float(os.environ.get("DP_MIN_IDLE", "30"))
MAX_IDLE_MSG_SESSION = int(os.environ.get("DP_MAX_IDLE_MSG", "3"))

# --- Sandbox ----------------------------------------------------------------
# Where tool/command execution runs: "local" | "docker" | "ssh".
SANDBOX_BACKEND = os.environ.get("DP_SANDBOX", "local")
CMD_TIMEOUT = int(os.environ.get("DP_CMD_TIMEOUT", "30"))
DOCKER_IMAGE = os.environ.get("DP_DOCKER_IMAGE", "python:3.12-slim")
SSH_TARGET = os.environ.get("DP_SSH_TARGET", "")          # e.g. user@host
SSH_REMOTE_DIR = os.environ.get("DP_SSH_REMOTE_DIR", "~/dp-workspace")

# --- Safety -----------------------------------------------------------------
# What to do with a high-impact action when there is no interactive console to
# approve it (e.g. running as a systemd service or behind the chat gateway):
#   "deny"    -> refuse it and tell the model to ask the owner directly (default)
#   "approve" -> auto-approve (UNATTENDED; only for fully trusted setups)
UNATTENDED_POLICY = os.environ.get("DP_UNATTENDED_POLICY", "deny").lower()

# --- Gateway (Hermes) -------------------------------------------------------
# TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USERS are read by the adapter itself.
GATEWAY_IDLE_TIMEOUT = int(os.environ.get("DP_GATEWAY_IDLE_TIMEOUT", "3600"))


def ensure_dirs() -> None:
    """Create the runtime data directories. Called lazily so that merely
    importing the package (e.g. during tests or `--help`) has no side effects
    on the filesystem until something actually needs to write."""
    for d in (MEMORY_DIR, WORKSPACE_DIR, VECTOR_DIR):
        d.mkdir(parents=True, exist_ok=True)
