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
#   openrouter/anthropic/claude-sonnet-4-6   (needs OPENROUTER_API_KEY)
#   ollama_chat/llama3           (needs Ollama running locally, no key)
CORE_MODEL = os.environ.get("DP_CORE_MODEL", "anthropic/claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("DP_MAX_TOKENS", "2048"))

# --- Model routing (optional) -----------------------------------------------
# CORE_MODEL is the FLOOR: every turn runs on it unless the router finds
# positive evidence that the turn needs more. Routing therefore only ever
# escalates upward, and buys quality rather than saving cost.
#
#   low  -> CORE_MODEL                       (always; also the fallback)
#   mid  -> DP_MID_MODEL, else low
#   top  -> DP_TOP_MODEL, else mid, else low
#
# Unset tiers collapse into the one below, so a two-tier setup is just
# DP_TOP_MODEL. DP_ROUTING=off (the default) pins everything to CORE_MODEL, so
# routing is strictly opt-in and an existing deployment is unaffected.
ROUTING = os.environ.get("DP_ROUTING", "off").strip().lower()
MID_MODEL = os.environ.get("DP_MID_MODEL", "").strip()
TOP_MODEL = os.environ.get("DP_TOP_MODEL", "").strip()

# Internal, non-conversational LLM jobs (consolidation, fact reconciliation).
# These need reliable strict JSON but no tools and no low latency, so they are
# pinned rather than routed. Defaults to CORE_MODEL: no behaviour change.
UTILITY_MODEL = os.environ.get("DP_UTILITY_MODEL", "").strip()

# Minimum cosine margin between the best and runner-up tier before an
# escalation is trusted. Measured on all-MiniLM-L6-v2, genuinely novel hard
# queries separate by only ~0.01-0.15, so a thin margin means "not confident"
# and must fall back DOWN rather than spend a flagship model on a coin flip.
ROUTE_MARGIN = float(os.environ.get("DP_ROUTE_MARGIN", "0.08"))


# --- Routing feedback -------------------------------------------------------
# Every routing decision is logged with its outcome, so the router's behaviour
# is inspectable rather than a black box (CLI: /routes). Logging is cheap and
# always on when routing is; it is the only way to tune the rest.
MAX_ROUTE_LOG = int(os.environ.get("DP_MAX_ROUTE_LOG", "2000"))

# Learning from that log is OFF by default, and deliberately so: it promotes
# real user phrasings into the exemplar bank, and a wrong promotion drags
# chitchat toward an escalation tier — degrading the floor it is meant to
# protect. Inspect /routes on real traffic first, then enable.
ROUTE_LEARNING = (os.environ.get("DP_ROUTE_LEARNING", "0").strip().lower()
                  not in ("", "0", "off", "false", "no"))
# Cap per tier, per user. Oldest learned exemplars are evicted first, so a bad
# promotion ages out instead of poisoning the bank forever.
MAX_LEARNED_EXEMPLARS = int(os.environ.get("DP_MAX_LEARNED", "40"))


def tier_model(tier: str) -> str:
    """Resolve a routing tier to a model string, collapsing unset tiers down."""
    if tier == "top":
        return TOP_MODEL or MID_MODEL or CORE_MODEL
    if tier == "mid":
        return MID_MODEL or CORE_MODEL
    return CORE_MODEL


def utility_model() -> str:
    return UTILITY_MODEL or CORE_MODEL

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
# Set these when CORE_MODEL uses a non-default API endpoint (e.g. Ollama Cloud).
# DP_API_KEY falls back to OLLAMA_CLOUD_API_KEY so the .env needs no change.
API_BASE = os.environ.get("DP_API_BASE")
API_KEY = (os.environ.get("DP_API_KEY")
           or os.environ.get("OLLAMA_CLOUD_API_KEY"))


def model_credentials(model: str) -> dict:
    """The ``api_base``/``api_key`` kwargs to call *model* with.

    DP_API_BASE/DP_API_KEY are a single global pair, so they describe exactly
    one endpoint: the one CORE_MODEL was pointed at. They are therefore applied
    ONLY to CORE_MODEL. Any other model resolves its own credentials from its
    provider's conventional env var (ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
    ...), which LiteLLM reads by itself when we pass None.

    This scoping is what makes a second model usable at all: passing the global
    override to every call would hand, say, an Ollama Cloud key and base URL to
    an OpenRouter model and break it.
    """
    if model == CORE_MODEL:
        return {"api_base": API_BASE or None, "api_key": API_KEY or None}
    return {"api_base": None, "api_key": None}

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

# --- Billing (usage pay gate) ------------------------------------------------
# Optional. Balance, transactions and per-message pricing all live in an
# external billing service, not here. When DP_BILLING_API_BASE is set, every
# gateway user (a real user_id, e.g. from Telegram) is checked against it
# before each LLM turn; local CLI / single-owner mode (user_id=None) is never
# gated. Unset = the gate is a no-op.
BILLING_API_BASE = os.environ.get("DP_BILLING_API_BASE", "").strip()
BILLING_CHECK_PATH = os.environ.get("DP_BILLING_CHECK_PATH", "/usage/check")
BILLING_API_KEY = os.environ.get("DP_BILLING_API_KEY", "")
BILLING_TIMEOUT = int(os.environ.get("DP_BILLING_TIMEOUT", "10"))
# Fallback top-up link shown (as a button, on adapters that support one) with
# the insufficient-balance message, when the check-usage response doesn't
# supply its own "payment_url". "{user_id}" is substituted if present, so one
# template can deep-link straight to that user's checkout. Unset = no link.
BILLING_TOPUP_URL = os.environ.get("DP_BILLING_TOPUP_URL", "").strip()


def ensure_dirs() -> None:
    """Create the runtime data directories. Called lazily so that merely
    importing the package (e.g. during tests or `--help`) has no side effects
    on the filesystem until something actually needs to write."""
    for d in (MEMORY_DIR, WORKSPACE_DIR, VECTOR_DIR):
        d.mkdir(parents=True, exist_ok=True)
