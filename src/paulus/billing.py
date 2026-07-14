"""Usage pay gate: ask an external billing service whether a gateway user may
receive another LLM turn, and block with a user-facing message when their
balance is exhausted or the service can't be reached.

Balance, transactions and per-message pricing are all owned by that external
service — this module only asks one question ("can this user get a reply
right now?") via a single check-usage call, and fails CLOSED on any error, so
a billing outage never turns into free, unmetered usage.

Entirely optional: leave ``DP_BILLING_API_BASE`` unset and the gate is a
no-op, so single-owner / CLI setups (``user_id=None``) are unaffected either
way.
"""
import json
import urllib.error
import urllib.request

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


def _check_usage(user_id) -> dict:
    """POST the usage-check request and return the parsed JSON body. Raises
    BillingError on any transport, HTTP, or parsing failure."""
    url = config.BILLING_API_BASE.rstrip("/") + config.BILLING_CHECK_PATH
    body = json.dumps({"user_id": str(user_id)}).encode()
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
        if data is not None and "allowed" in data:
            return data
        raise BillingError(f"usage check HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise BillingError(f"usage check request failed: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BillingError(f"usage check returned invalid JSON: {exc}") from exc
    if "allowed" not in data:
        raise BillingError(f"usage check response missing 'allowed': {data!r}")
    return data


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
    message = data.get("message") or INSUFFICIENT_BALANCE_MESSAGE
    url = data.get("payment_url") or _topup_url(user_id)
    if url:
        message = f"{message}{_PAY_LINK_MARKER}{url}"
    return False, message
