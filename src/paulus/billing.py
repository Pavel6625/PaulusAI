"""Usage-based billing: gate each gateway turn on the user's balance, then
meter what the turn actually costs, per token, as reported by the provider.

Two calls to an external billing service, which owns balances, transactions,
markup and currency:

* **check** (``DP_BILLING_CHECK_PATH``) — before a turn: "may this user get a
  reply right now?" (gate()).
* **record** (``DP_BILLING_RECORD_PATH``) — after every LLM call made inside a
  metered turn: the call's raw USD cost and token counts (record()). The cost
  is OpenRouter's own ``usage.cost`` when available (llm.py extracts it), so
  the user pays for the tokens the turn really used, not a flat per-message
  price. A long tool loop costs more than "hi", as it should.

A turn is metered by running it inside ``with metered(user_id):`` — agent
respond() and background tasks do; bot-initiated work (idle check-ins, memory
consolidation) does not, so it is never billed to the user. Subagent threads
inherit the turn's meter (they run in a copy of its context).

If a record reply says the balance is exhausted (``"allowed": false``), or the
record call fails, the meter is stopped: the call that was just made is kept
(its reply already exists), but the *next* LLM call in that turn raises
BalanceExhausted instead of spending more. A turn can therefore overrun the
balance by at most the calls in flight; the service settles that (e.g. to 0).

OpenRouter calls are priced from OpenRouter's own generation record, which
takes a moment to appear after the reply, so they are recorded in the
background (record_deferred) rather than holding the reply back. Their
balance check lands a little later — the overrun above can grow by the calls
made meanwhile. flush() waits for those records (the gateway calls it on
shutdown).

Fails CLOSED throughout, so a billing outage never turns into free, unmetered
usage. Entirely optional: leave ``DP_BILLING_API_BASE`` unset and every entry
point is a no-op, so single-owner / CLI setups (``user_id=None``) are
unaffected either way.
"""
import contextlib
import contextvars
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import config, security

INSUFFICIENT_BALANCE_MESSAGE = (
    "You're out of balance. Please top up to keep chatting."
)
SERVICE_UNAVAILABLE_MESSAGE = (
    "Sorry — I can't verify your balance right now, so I can't respond. "
    "Please try again shortly."
)
TOPUP_BUTTON_LABEL = "\U0001f4b3 Top up"

# A payment link is carried inline in the block message, on its own trailing
# line after this marker, so gate() keeps returning a single string (no API
# change for existing callers like the CLI) while the gateway can still pull
# the URL back out to render a real button. Never shown to the user as-is —
# split_pay_link() strips it before anything is displayed.
_PAY_LINK_MARKER = "\n\n[[PAULUS_PAY_LINK]] "


class BillingError(Exception):
    """Raised internally when the usage-check call fails; always caught by gate()."""


def enabled() -> bool:
    return bool(config.BILLING_API_BASE)


def split_pay_link(text: str) -> tuple[str, str | None]:
    """Split an embedded payment link (added by gate()) off the visible text.
    Returns ``(clean_text, url)`` — ``url`` is None when no link is present."""
    idx = text.find(_PAY_LINK_MARKER)
    if idx == -1:
        return text, None
    return text[:idx], text[idx + len(_PAY_LINK_MARKER):].strip()


def _topup_url(user_id) -> str | None:
    template = config.BILLING_TOPUP_URL
    if not template:
        return None
    return template.replace("{user_id}", str(user_id))


def _post(path, payload, what) -> dict:
    """POST *payload* as JSON to the billing service and return the parsed JSON
    body (``{}`` for an empty 2xx body). Raises BillingError on any transport,
    HTTP, or parsing failure. *what* names the call in error messages."""
    url = config.BILLING_API_BASE.rstrip("/") + path
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if config.BILLING_API_KEY:
        headers["Authorization"] = f"Bearer {config.BILLING_API_KEY}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=config.BILLING_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # A non-2xx status (e.g. 402 Payment Required) may still carry a JSON
        # body reporting {"allowed": false, ...} rather than a real service
        # failure — honour that body if present before giving up on the call.
        try:
            data = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            data = None
        if isinstance(data, dict) and "allowed" in data:
            return data
        raise BillingError(f"{what} HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise BillingError(f"{what} request failed: {exc}") from exc
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BillingError(f"{what} returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BillingError(f"{what} returned a non-object: {data!r}")
    return data


def _check_usage(user_id) -> dict:
    """POST the usage-check request and return the parsed JSON body. Raises
    BillingError on any transport, HTTP, or parsing failure."""
    data = _post(config.BILLING_CHECK_PATH, {"user_id": str(user_id)}, "usage check")
    if "allowed" not in data:
        raise BillingError(f"usage check response missing 'allowed': {data!r}")
    return data


def _block_message(user_id, data) -> str:
    """The insufficient-balance message for a service reply, with its top-up
    link embedded (see split_pay_link) when one is known."""
    message = data.get("message") or INSUFFICIENT_BALANCE_MESSAGE
    url = data.get("payment_url") or _topup_url(user_id)
    if url:
        message = f"{message}{_PAY_LINK_MARKER}{url}"
    return message


def gate(user_id) -> tuple[bool, str | None]:
    """Check whether *user_id* may receive another LLM turn.

    Returns ``(allowed, block_message)`` — ``block_message`` is None when
    allowed. Always allowed when billing isn't configured
    (``DP_BILLING_API_BASE`` unset) or *user_id* is None (CLI / single-owner
    mode). Fails CLOSED: any error talking to the billing service blocks the
    turn rather than granting free usage.
    """
    if not enabled() or user_id is None:
        return True, None

    try:
        data = _check_usage(user_id)
    except BillingError as exc:
        security.audit("billing_check_error", f"{user_id}: {exc}")
        return False, SERVICE_UNAVAILABLE_MESSAGE

    if data.get("allowed"):
        return True, None

    security.audit("billing_blocked", f"{user_id} balance={data.get('balance')}")
    return False, _block_message(user_id, data)


# ---------------------------------------------------------------------------
# Per-token metering
# ---------------------------------------------------------------------------

class BalanceExhausted(Exception):
    """Raised (by llm.py, via ensure_can_spend) before an LLM call in a metered
    turn whose meter has been stopped. ``block_message`` is what gate() would
    return — the pay link embedded — for callers that deliver it as the reply;
    ``str(exc)`` is the clean, user-visible text."""

    def __init__(self, block_message):
        self.block_message = block_message
        super().__init__(split_pay_link(block_message)[0])


@dataclass
class Meter:
    """Running totals for one metered turn or background task. Shared by the
    subagent threads the turn fans out to, hence the lock."""
    user_id: str
    kind: str = "turn"
    calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_sources: dict = field(default_factory=dict)
    block_message: str | None = None     # set once the meter is stopped
    pending: int = 0                     # deferred records not yet sent
    closed: bool = False                 # the metered block has exited
    summarized: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_meter: contextvars.ContextVar[Meter | None] = contextvars.ContextVar(
    "paulus_billing_meter", default=None)

# Background threads still sending a deferred record (see flush()).
_inflight: set[threading.Thread] = set()
_inflight_lock = threading.Lock()


@contextlib.contextmanager
def metered(user_id, kind="turn"):
    """Meter every LLM call made in this block (and in subagent threads it
    spawns) against *user_id*. Yields the Meter, or None — a no-op — when
    billing is off or *user_id* is None. *kind* ("turn", "task") is passed to
    the billing service with each record."""
    if not enabled() or user_id is None:
        yield None
        return
    meter = Meter(user_id=str(user_id), kind=kind)
    token = _meter.set(meter)
    try:
        yield meter
    finally:
        _meter.reset(token)
        with meter._lock:
            meter.closed = True
        _summarize(meter)


def _summarize(meter):
    """Audit the turn's totals once: when its block has exited AND its last
    deferred record has landed, so the summary includes every call's cost."""
    with meter._lock:
        if not meter.closed or meter.pending or meter.summarized or not meter.calls:
            return
        meter.summarized = True
        line = (f"{meter.user_id} {meter.kind} calls={meter.calls} "
                f"tokens={meter.prompt_tokens}+{meter.completion_tokens} "
                f"cost_usd={meter.cost_usd:.6f} "
                f"sources={json.dumps(meter.cost_sources, sort_keys=True)}")
    security.audit("billing_usage", line)


def metering() -> bool:
    """Whether the current context is inside a metered turn — llm.py asks
    before doing any work to extract usage."""
    return _meter.get() is not None


def ensure_can_spend() -> None:
    """Raise BalanceExhausted if the current turn's meter has been stopped.
    Called by llm.py before every model call; a no-op outside a metered turn."""
    meter = _meter.get()
    if meter is not None and meter.block_message is not None:
        raise BalanceExhausted(meter.block_message)


def _stop(meter, message):
    with meter._lock:
        if meter.block_message is None:
            meter.block_message = message


def record(*, cost_usd, cost_source, prompt_tokens, completion_tokens,
           model, request_id=None) -> None:
    """Report one LLM call's usage to the billing service, if the current
    context is metered. Never raises: the call has already happened.

    *cost_usd* is the raw provider cost (OpenRouter's ``usage.cost``) before
    any markup — that's the service's job. *cost_source* says where it came
    from: ``"provider"`` (reported by the provider), ``"estimate"`` (LiteLLM's
    price map applied to the token counts) or ``"none"`` (unpriced; only the
    token counts are sent). *request_id* is the provider's generation id,
    usable by the service as an idempotency key."""
    meter = _meter.get()
    if meter is None:
        return
    _send(meter, cost_usd=cost_usd, cost_source=cost_source,
          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
          model=model, request_id=request_id)


def record_deferred(resolve, *, prompt_tokens, completion_tokens, model,
                    request_id=None) -> None:
    """Like record(), but the cost is looked up off the calling thread: the
    reply isn't held back while ``resolve()`` (which may wait on a provider,
    e.g. OpenRouter's generation record) runs. ``resolve()`` returns
    ``(cost_usd, cost_source)``; whatever it raises is audited and the call
    recorded as unpriced, never lost. A no-op outside a metered turn."""
    meter = _meter.get()
    if meter is None:
        return
    with meter._lock:
        meter.pending += 1

    def run():
        try:
            try:
                cost_usd, cost_source = resolve()
            except Exception as exc:
                security.audit("billing_cost_error", f"{meter.user_id} {model} "
                               f"{request_id}: {exc}")
                cost_usd, cost_source = 0.0, "none"
            _send(meter, cost_usd=cost_usd, cost_source=cost_source,
                  prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                  model=model, request_id=request_id)
        finally:
            with meter._lock:
                meter.pending -= 1
            _summarize(meter)
            with _inflight_lock:
                _inflight.discard(threading.current_thread())

    thread = threading.Thread(target=run, name="paulus-billing", daemon=True)
    with _inflight_lock:
        _inflight.add(thread)
    thread.start()


def flush(timeout=None) -> None:
    """Wait for deferred records still being sent (gateway shutdown, tests).
    *timeout* bounds the whole wait, in seconds; None waits for all."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        with _inflight_lock:
            threads = list(_inflight)
        if not threads:
            return
        for thread in threads:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return
            thread.join(remaining)


def _send(meter, *, cost_usd, cost_source, prompt_tokens, completion_tokens,
          model, request_id):
    """Count one call on *meter* and POST its record (see record())."""
    with meter._lock:
        meter.calls += 1
        meter.cost_usd += cost_usd
        meter.prompt_tokens += prompt_tokens
        meter.completion_tokens += completion_tokens
        meter.cost_sources[cost_source] = meter.cost_sources.get(cost_source, 0) + 1
    payload = {
        "user_id": meter.user_id,
        "kind": meter.kind,
        "model": model,
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
    }
    if cost_source == "none":
        security.audit("billing_unpriced", f"{meter.user_id} {model} "
                       f"tokens={prompt_tokens}+{completion_tokens}")
    try:
        data = _post(config.BILLING_RECORD_PATH, payload, "usage record")
    except BillingError as exc:
        # Unrecorded usage is free usage: keep the full payload in the audit
        # log for reconciliation, and spend nothing more this turn.
        security.audit("billing_record_error", f"{exc} :: {json.dumps(payload)}")
        _stop(meter, SERVICE_UNAVAILABLE_MESSAGE)
        return
    if data.get("allowed") is False:
        security.audit("billing_exhausted",
                       f"{meter.user_id} balance={data.get('balance')}")
        _stop(meter, _block_message(meter.user_id, data))
