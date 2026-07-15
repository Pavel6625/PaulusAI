"""Per-turn model routing: choose ONE model for one conversational turn.

CORE_MODEL is the floor. A turn escalates to a stronger tier only on positive
evidence that it needs one, so a misfire costs quality-as-usual rather than a
flagship bill, and a router that cannot run at all degrades to exactly the
behaviour of a deployment with no routing.

Two signals, deliberately in this order:

1. Structural (deterministic). A traceback, a code fence or an attached
   document is near-proof that a turn is not chitchat. This is regex, not
   inference, so it is high-precision and free.

2. Semantic (nearest-class over embedded exemplars). Handles the phrasings a
   regex cannot enumerate.

The order matters because the two fail in *opposite* places. Measured on
all-MiniLM-L6-v2, a raw traceback embeds no closer to "debug this stack trace"
(0.190) than "good morning" embeds to "help me plan a trip" (0.318): short-text
embeddings capture topic and register, not difficulty. The structural pass
catches exactly the cases where the semantic pass is weakest.

Semantic escalations are additionally gated on a margin (config.ROUTE_MARGIN):
novel hard queries beat the runner-up tier by as little as 0.006, and a
coin-flip is not evidence. Below the margin the turn falls back DOWN a tier.

Routing never fails a turn: any problem (embeddings absent, model not
configured, unknown tier) yields CORE_MODEL.
"""
import datetime
import json
import math
import re
import uuid
from pathlib import Path

from . import config

AVAILABLE = False
_embed_fn = None
_bank = None          # tier -> list[normalised vector]   (built-in, shared)
_learned_cache = {}   # uid -> {tier: [(text, vector)]}   (per user, from disk)

# The turn each user was last routed for, so a complaint on the NEXT turn can
# be attributed to the decision that caused it. In-process only: a restart
# loses at most one pending label per user, which is acceptable for a signal
# that is advisory rather than load-bearing.
_last_turn = {}

# Ordered weakest -> strongest; index doubles as the tier's rank.
TIERS = ("low", "mid", "top")

# The agent's own memory and skill tools. The system prompt tells the model to
# `recall` before claiming ignorance and to `find_skill` for anything non-trivial,
# so these fire constantly on small talk and carry no difficulty signal. Every
# other tool reaches outside the agent's head, which does. Defined as an
# exclusion so a newly added tool counts as effort by default.
INTROSPECTIVE_TOOLS = {"remember", "recall", "find_skill", "save_skill"}


# ---------------------------------------------------------------------------
# Exemplars
# ---------------------------------------------------------------------------
# Phrasings, not topics. "low" must be populated: without it, chitchat is
# classified only by its distance to mid/top, and "good morning" then scores a
# higher escalation similarity than an actual stack trace.
EXEMPLARS = {
    "low": [
        "hey", "hi there", "good morning", "goodnight", "thanks!",
        "thank you so much", "how are you", "how was your day", "ok cool",
        "sounds good", "haha nice", "i'm tired", "i had a rough day",
        "that's great news", "miss you", "what did i tell you about my family",
        "remind me what we talked about", "see you later", "yeah exactly",
    ],
    "mid": [
        "summarise this article for me",
        "write an email about the broken heating",
        "what is the difference between these two options",
        "help me plan a trip", "translate this paragraph into German",
        "explain how DNS works", "draft a polite reply to this message",
        "write a formal letter", "give me a recipe for dinner",
    ],
    "top": [
        "debug this stack trace and tell me why it fails",
        "review this function for correctness bugs",
        "why does this code deadlock",
        "design the architecture for this system",
        "prove this statement step by step",
        "analyse this contract and flag the risks",
        "refactor this module and explain the tradeoffs",
        "work through this problem carefully step by step",
        "why is this returning the wrong value",
    ],
}

# Near-proof of a non-chitchat turn. Kept narrow on purpose: a false positive
# here spends a flagship model on small talk.
_STRUCTURAL = re.compile(
    r"```"
    r"|Traceback \(most recent call last\)"
    r'|^\s*File "[^"]+", line \d+'
    r"|\b[A-Za-z_]+Error\b|\bException\b|\bstack trace\b"
    r"|^\s*(def|class|import|from|SELECT|function|const|public)\s",
    re.I | re.M,
)

# An input this long is not small talk, and the weakest tier tends to lose the
# thread of it regardless of topic.
_LONG_INPUT = 400


def _norm(vec):
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else list(vec)


def _cos(a, b):
    # strict: both vectors come from the same embedder, so a length mismatch is
    # a bug worth surfacing rather than silently truncating to a wrong score.
    return sum(x * y for x, y in zip(a, b, strict=True))


def init():
    """Bring the embedder up. On any failure stay unavailable, which makes
    route() fall through to CORE_MODEL rather than raise. Mirrors
    vectorstore.init() — embeddings are an optional extra."""
    global AVAILABLE, _embed_fn, _bank
    if config.ROUTING == "off":
        return False
    try:
        from chromadb.utils import embedding_functions
        _embed_fn = embedding_functions.DefaultEmbeddingFunction()
        _bank = {
            tier: [_norm(v) for v in _embed_fn(list(phrases))]
            for tier, phrases in EXEMPLARS.items()
        }
        AVAILABLE = True
    except Exception as e:
        AVAILABLE = False
        print(f"[router] embeddings unavailable, routing disabled "
              f"(every turn uses the core model): {e}")
    return AVAILABLE


def _similarities(text, user_id=None):
    """tier -> best cosine similarity between *text* and that tier's exemplars,
    over the built-in bank plus anything learned for this user."""
    q = _norm(_embed_fn([text])[0])
    learned = _learned_vectors(user_id)
    out = {}
    for tier, vecs in _bank.items():
        scores = [_cos(q, e) for e in vecs]
        scores += [_cos(q, v) for _t, v in learned.get(tier, [])]
        out[tier] = max(scores)
    return out


def classify(text, user_id=None, has_images=False, has_documents=False):
    """Decide a tier for *text*. Returns (tier, reason) — reason is for the
    decision log and the /route command, and is why a choice can be explained
    to a human without re-running the model."""
    if config.ROUTING == "off":
        return "low", "routing off"

    # Vision is a hard constraint, not a preference: the floor may be blind.
    if has_images:
        return _vision_tier(), "image attached"

    if _STRUCTURAL.search(text or ""):
        return "top", "structural: code/traceback"
    if has_documents:
        return "mid", "structural: document attached"
    if len(text or "") > _LONG_INPUT:
        return "mid", f"structural: long input ({len(text)} chars)"

    if not AVAILABLE:
        return "low", "embeddings unavailable"

    try:
        sims = _similarities(text, user_id)
    except Exception as e:
        return "low", f"embedding failed: {e}"

    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (runner, runner_score) = ranked[0], ranked[1]
    margin = best_score - runner_score

    # An unconfident escalation is not evidence — take the weaker of the two
    # candidates. Ties and near-ties therefore settle on the floor.
    if margin < config.ROUTE_MARGIN:
        chosen = min((best, runner), key=TIERS.index)
        return chosen, (f"low margin {margin:.3f} between {best}/{runner}"
                        f" -> {chosen}")
    return best, f"semantic {best} ({best_score:.3f}, margin {margin:.3f})"


def _vision_tier():
    """The weakest configured tier that can actually see."""
    from . import llm
    for tier in TIERS:
        if llm.supports_vision(config.tier_model(tier)):
            return tier
    return "low"      # nothing catalogued as sighted; let the call try anyway


def vision_model():
    """The model an image turn would be routed to. The gateway asks this before
    accepting an image, so its up-front check matches what route() will pick."""
    if config.ROUTING == "off":
        return config.CORE_MODEL
    return config.tier_model(_vision_tier())


def route(text, user_id=None, has_images=False, has_documents=False):
    """The model to use for this turn. Returns (model, tier, reason)."""
    tier, reason = classify(text, user_id=user_id, has_images=has_images,
                            has_documents=has_documents)
    return config.tier_model(tier), tier, reason


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------
# Per user, mirroring memory.py: routing records a person's raw phrasings, and
# the rest of this codebase keeps one user's text out of another user's reach.
# It also makes learning personal, which is the more accurate design anyway —
# people phrase the same need differently.

def _user_dir(user_id):
    if user_id is None:
        return None
    return config.MEMORY_DIR / "users" / config.safe_uid(user_id)


def _route_log(user_id=None) -> Path:
    d = _user_dir(user_id)
    return d / "routing.jsonl" if d else config.MEMORY_DIR / "routing.jsonl"


def _exemplar_file(user_id=None) -> Path:
    d = _user_dir(user_id)
    return (d / "route_exemplars.json" if d
            else config.MEMORY_DIR / "route_exemplars.json")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _append(user_id, record):
    """Append one event. Best-effort: the log is diagnostics, and must never be
    the reason a turn fails."""
    try:
        path = _route_log(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        _trim(path)
    except Exception as e:
        print(f"[router] could not write decision log: {e}")


def _trim(path):
    if config.MAX_ROUTE_LOG <= 0:
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > config.MAX_ROUTE_LOG:
        keep = lines[-config.MAX_ROUTE_LOG:]
        path.write_text("\n".join(keep) + "\n", encoding="utf-8")


def read_log(user_id=None):
    try:
        raw = _route_log(user_id).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    out = []
    for line in raw.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a torn final line must not lose the whole log
    return out


def log_decision(text, tier, reason, user_id=None):
    """Record a routing decision and return its turn id. Logged at decision
    time, not turn end, so a turn that crashes still leaves evidence."""
    turn_id = uuid.uuid4().hex[:12]
    _append(user_id, {
        "type": "decision", "id": turn_id, "ts": _now(),
        "text": (text or "")[:500], "tier": tier, "reason": reason,
    })
    _last_turn[config.safe_uid(user_id)] = turn_id
    return turn_id


def log_outcome(turn_id, tools_used, user_id=None):
    """Record what the turn actually needed. Reaching outside its own memory is
    objective evidence the turn was not chitchat — unlike a complaint, it cannot
    be triggered by an unlucky choice of words."""
    effort = sorted(set(tools_used) - INTROSPECTIVE_TOOLS)
    if not effort:
        return
    _append(user_id, {"type": "outcome", "ref": turn_id, "ts": _now(),
                      "effort_tools": effort})


def log_complaint(user_id=None):
    """Attribute a complaint to the PREVIOUS turn: the owner is reacting to the
    reply that turn produced, not to the message they just sent."""
    turn_id = _last_turn.get(config.safe_uid(user_id))
    if turn_id is None:
        return
    _append(user_id, {"type": "complaint", "ref": turn_id, "ts": _now()})


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def _load_state(user_id):
    """Learned exemplars plus the ids already considered.

    ``seen`` is what makes learning incremental, and it is not optional: without
    it, learn() re-scans the whole log every run and an exemplar evicted by the
    cap is immediately re-added by the same log entry that first promoted it, so
    the bank churns instead of ageing."""
    try:
        data = json.loads(_exemplar_file(user_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"seen": [], "tiers": {}}
    return {
        "seen": list(data.get("seen", [])),
        "tiers": {t: list(v) for t, v in data.get("tiers", {}).items()
                  if t in TIERS},
    }


def _save_state(user_id, state):
    path = _exemplar_file(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bounded by the log's own cap: once a turn has aged out of the log it can
    # never be seen again, so remembering it forever buys nothing.
    if config.MAX_ROUTE_LOG > 0:
        state["seen"] = state["seen"][-config.MAX_ROUTE_LOG:]
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _learned_cache.pop(config.safe_uid(user_id), None)   # force a re-embed


def _load_learned(user_id):
    return _load_state(user_id)["tiers"]


def _learned_vectors(user_id):
    """Learned exemplars for *user_id*, embedded and cached."""
    if not AVAILABLE:
        return {}
    uid = config.safe_uid(user_id)
    if uid in _learned_cache:
        return _learned_cache[uid]
    learned = _load_learned(user_id)
    banked = {}
    for tier, phrases in learned.items():
        if phrases:
            banked[tier] = list(zip(phrases, [_norm(v) for v in _embed_fn(phrases)],
                                    strict=True))
    _learned_cache[uid] = banked
    return banked


def _promote(tier):
    """One step up, never two. The evidence says 'this needed more than it got',
    not 'this needed the flagship'."""
    i = TIERS.index(tier)
    return TIERS[min(i + 1, len(TIERS) - 1)]


def learn(user_id=None):
    """Fold the decision log's outcomes back into this user's exemplars.

    Purely mechanical — no model call. A turn that was routed low but then had
    to reach outside its own memory (or drew a complaint) is a labelled example
    of an under-route, and its phrasing is exactly what the semantic pass failed
    to recognise. Measured separation is thin enough that near-paraphrases are
    what it matches on, so feeding it the real wording is the medicine.

    Returns the number of exemplars added.
    """
    if not config.ROUTE_LEARNING:
        return 0

    log = read_log(user_id)
    decisions = {r["id"]: r for r in log if r.get("type") == "decision"}

    state = _load_state(user_id)
    learned, seen = state["tiers"], state["seen"]
    seen_set = set(seen)
    existing = {p for ps in learned.values() for p in ps}
    added = 0

    # Log order, not set order: learning must be reproducible, and the cap
    # evicts oldest-first, which is only meaningful if "oldest" is well defined.
    for record in log:
        if record.get("type") not in ("outcome", "complaint"):
            continue
        turn_id = record.get("ref")
        if turn_id in seen_set:
            continue
        seen.append(turn_id)
        seen_set.add(turn_id)

        d = decisions.get(turn_id)
        if d is None or d["tier"] == "top":
            continue
        # Only the semantic pass can be wrong in a way exemplars fix. A
        # structural or vision decision was already made on hard evidence.
        if not d.get("reason", "").startswith(("semantic", "low margin")):
            continue
        text = (d.get("text") or "").strip()
        if not text or text in existing:
            continue
        bucket = learned.setdefault(_promote(d["tier"]), [])
        bucket.append(text)
        existing.add(text)
        added += 1
        cap = config.MAX_LEARNED_EXEMPLARS
        if cap > 0 and len(bucket) > cap:
            del bucket[:len(bucket) - cap]          # evict oldest first

    _save_state(user_id, state)
    return added


def learned_summary(user_id=None):
    """Human-readable view of what routing has learned, for /routes."""
    learned = _load_learned(user_id)
    if not any(learned.values()):
        return "(nothing learned yet)"
    lines = []
    for tier in TIERS:
        for phrase in learned.get(tier, []):
            lines.append(f"  [{tier}] {phrase[:70]}")
    return "\n".join(lines)
