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
import re
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
PRESENCE_FILE = MEMORY_DIR / "presence.json"  # per-user idle/nudge state

# --- Per-user isolation -----------------------------------------------------
# A user_id arrives from the chat gateway and is used as a filesystem path
# component (for both memory and the sandbox workspace), so it MUST be
# sanitised — a value like "../other" would otherwise escape the user's dir.
_SAFE_UID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def safe_uid(user_id) -> str:
    """Sanitise a user_id into a single safe path component."""
    return _SAFE_UID_RE.sub("_", str(user_id))


def user_workspace(user_id=None) -> Path:
    """The workspace root for *user_id*. ``None`` (CLI / single-user mode)
    falls back to the shared WORKSPACE_DIR so existing setups are unaffected;
    a concrete id is isolated under ``WORKSPACE_DIR/users/<safe_uid>/``."""
    if user_id is None:
        return WORKSPACE_DIR
    return WORKSPACE_DIR / "users" / safe_uid(user_id)

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


def _parse_quiet(spec):
    """Parse "START-END" (24h hours, local time) into (start, end) ints.
    Returns (None, None) when unset or malformed (i.e. no quiet window)."""
    if not spec or "-" not in spec:
        return None, None
    try:
        start, end = (int(x) % 24 for x in spec.split("-", 1))
    except ValueError:
        return None, None
    return start, end


# Local-time window during which the agent stays silent (no proactive nudges).
# e.g. DP_QUIET_HOURS=23-7 mutes 23:00 up to (not including) 07:00. Empty = off.
QUIET_START, QUIET_END = _parse_quiet(os.environ.get("DP_QUIET_HOURS", ""))


def in_quiet_hours(now=None):
    """True if local time falls inside the configured quiet window."""
    if QUIET_START is None or QUIET_END is None or QUIET_START == QUIET_END:
        return False
    from datetime import datetime
    hour = (now or datetime.now()).hour
    if QUIET_START < QUIET_END:
        return QUIET_START <= hour < QUIET_END
    return hour >= QUIET_START or hour < QUIET_END   # window wraps midnight

# --- Sandbox ----------------------------------------------------------------
# Where tool/command execution runs: "local" | "docker" | "ssh".
SANDBOX_BACKEND = os.environ.get("DP_SANDBOX", "local")
CMD_TIMEOUT = int(os.environ.get("DP_CMD_TIMEOUT", "30"))
DOCKER_IMAGE = os.environ.get("DP_DOCKER_IMAGE", "python:3.12-slim")
SSH_TARGET = os.environ.get("DP_SSH_TARGET", "")          # e.g. user@host
SSH_REMOTE_DIR = os.environ.get("DP_SSH_REMOTE_DIR", "~/dp-workspace")

# --- Web access -------------------------------------------------------------
# `web_search` works with no key via DuckDuckGo's HTML endpoint. Set a provider
# key (TAVILY_API_KEY or BRAVE_API_KEY) for a higher-quality keyed backend;
# DP_SEARCH_PROVIDER pins one explicitly ("duckduckgo"|"tavily"|"brave"), while
# "auto" (default) picks a keyed provider when its key is present, else DuckDuckGo.
SEARCH_PROVIDER = os.environ.get("DP_SEARCH_PROVIDER", "auto")
WEB_MAX_RESULTS = int(os.environ.get("DP_WEB_MAX_RESULTS", "5"))
WEB_MAX_CHARS = int(os.environ.get("DP_WEB_MAX_CHARS", "8000"))   # per fetched page
WEB_MAX_BYTES = int(os.environ.get("DP_WEB_MAX_BYTES", str(5 * 1024 * 1024)))
WEB_TIMEOUT = int(os.environ.get("DP_WEB_TIMEOUT", "20"))
WEB_USER_AGENT = os.environ.get(
    "DP_WEB_USER_AGENT",
    "Mozilla/5.0 (compatible; PaulusAI/1.0; +https://github.com/Pavel6625/PaulusAI)",
)

# --- Safety -----------------------------------------------------------------
# What to do with a high-impact action when there is no interactive console to
# approve it (e.g. running as a systemd service or behind the chat gateway):
#   "deny"    -> refuse it and tell the model to ask the owner directly (default)
#   "approve" -> auto-approve (UNATTENDED; only for fully trusted setups)
UNATTENDED_POLICY = os.environ.get("DP_UNATTENDED_POLICY", "deny").lower()

# Let a reachable, trusted gateway user approve a high-impact action interactively
# (e.g. inline Approve/Deny buttons on Telegram) instead of falling straight
# through to UNATTENDED_POLICY. The unattended policy still applies whenever the
# user can't be reached. On by default; set DP_GATEWAY_APPROVALS=0 to disable.
GATEWAY_APPROVALS = (
    os.environ.get("DP_GATEWAY_APPROVALS", "1").strip().lower()
    not in ("", "0", "off", "false", "no")
)
# How long to wait for that interactive answer before failing safe (DENY).
APPROVAL_TIMEOUT = int(os.environ.get("DP_APPROVAL_TIMEOUT", "300"))

# --- Gateway (Hermes) -------------------------------------------------------
# TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USERS are read by the adapter itself.
GATEWAY_IDLE_TIMEOUT = int(os.environ.get("DP_GATEWAY_IDLE_TIMEOUT", "3600"))

# --- Payments (TON Connect) -------------------------------------------------
# PaulusAI is a STATELESS payment validator. The Django backend owns balances
# and transactions; when its mini app reports a payment, Django calls our
# validate endpoint, we read the transaction from a public TON API and return
# the on-chain facts (amount, sender, comment). Django decides the credit and
# enforces idempotency (UNIQUE tx hash). We never write to Django and keep no
# payment state here.
#
# The HTTP endpoint needs the `payments` extra (Flask); the verification logic
# uses only the stdlib. A transaction validates only when ALL hold: it succeeded
# on-chain, its destination is our wallet, it carries a positive value, and it
# is not stale. The reported amount is read FROM THE CHAIN, never from the caller.
TON_API_BASE = os.environ.get("DP_TON_API_BASE", "https://tonapi.io/v2").rstrip("/")
TON_API_KEY = os.environ.get("DP_TON_API_KEY", "")          # optional; raises rate limits
# Our receiving wallet. Set it in the same raw form the API returns
# ("0:<64-hex>") for an exact match; friendly "EQ..."/"UQ..." forms are compared
# best-effort. A native-TON payment whose destination isn't this wallet is rejected.
TON_WALLET_ADDRESS = os.environ.get("DP_TON_WALLET_ADDRESS", "")
# Our USD₮ jetton wallet (the jetton wallet contract that holds OUR USD₮). A
# stablecoin payment is accepted only if its transfer-notification comes from
# this address — that proves both "it's USD₮" and "it's to us". Leave empty to
# accept native TON only. Set it in raw "0:<hex>" form.
USDT_JETTON_WALLET = os.environ.get("DP_USDT_JETTON_WALLET", "")
# Reject payments whose on-chain timestamp is older than this (seconds); guards
# against someone replaying an ancient unrelated transaction. 0 disables.
TON_MAX_TX_AGE = int(os.environ.get("DP_TON_MAX_TX_AGE", "3600"))
TON_HTTP_TIMEOUT = int(os.environ.get("DP_TON_HTTP_TIMEOUT", "20"))
# How many recent wallet transactions a memo scan (find_payment) inspects.
TON_SCAN_LIMIT = int(os.environ.get("DP_TON_SCAN_LIMIT", "100"))
# Shared secret the Django backend must present (Bearer) to call the validate
# endpoint. Empty disables the check (dev only) — set it in production.
PAYMENTS_TOKEN = os.environ.get("DP_PAYMENTS_TOKEN", "")

# --- Payment gate (agent -> Django) -----------------------------------------
# The read/spend side: the gateway checks a user's balance before answering and
# debits usage after. Gating is INACTIVE unless both URL and token are set, so
# the CLI and unpaid deployments are unaffected.
#   DP_PAYMENTS_BACKEND_URL -> the Django API root, e.g. https://api.example.com/api
#   DP_PAYMENTS_BACKEND_TOKEN -> matches the backend's INTERNAL_API_TOKEN
PAYMENTS_BACKEND_URL = os.environ.get("DP_PAYMENTS_BACKEND_URL", "").rstrip("/")
PAYMENTS_BACKEND_TOKEN = os.environ.get("DP_PAYMENTS_BACKEND_TOKEN", "")
PAYMENTS_HTTP_TIMEOUT = int(os.environ.get("DP_PAYMENTS_HTTP_TIMEOUT", "15"))
# USD charged per user interaction. 0 gates on balance but never debits.
USAGE_COST_USD = float(os.environ.get("DP_USAGE_COST_USD", "0.02"))
# URL the "top up" button opens (the Mini App, e.g. https://t.me/YourBot/pay).
PAYMENTS_MINIAPP_URL = os.environ.get("DP_PAYMENTS_MINIAPP_URL", "")
# On a backend outage: allow turns (default, fail-open) or refuse them.
PAYMENTS_FAIL_CLOSED = (
    os.environ.get("DP_PAYMENTS_FAIL_CLOSED", "0").strip().lower()
    not in ("", "0", "off", "false", "no")
)


def ensure_dirs() -> None:
    """Create the runtime data directories. Called lazily so that merely
    importing the package (e.g. during tests or `--help`) has no side effects
    on the filesystem until something actually needs to write."""
    for d in (MEMORY_DIR, WORKSPACE_DIR, VECTOR_DIR):
        d.mkdir(parents=True, exist_ok=True)
