"""The usage pay-gate.

Before the agent answers a gateway user, we check their balance; if it can't
cover one interaction we return a friendly top-up prompt instead of running the
model. After a successful answer we debit the per-use cost.

Design choices:
  * Opt-in — inactive unless a backend URL and token are configured, so the CLI
    and unpaid deployments are entirely unaffected.
  * Fail-open — if the backend is unreachable we let the turn through rather than
    locking a paying user out during an outage (logged for follow-up). Flip
    DP_PAYMENTS_FAIL_CLOSED=1 to refuse instead.
"""
import logging

from .. import config, security
from . import backend

log = logging.getLogger(__name__)


def enabled():
    """Payments gating is active only when the backend is configured."""
    return bool(config.PAYMENTS_BACKEND_URL and config.PAYMENTS_BACKEND_TOKEN)


def _client():
    return backend.BackendClient()


def pay_prompt(balance=None):
    """The message shown when a user is out of funds."""
    lead = "You're out of credit for PaulusAI."
    if balance is not None and balance > 0:
        lead = (f"Your balance (${balance:.2f}) is too low to continue "
                "using PaulusAI.")
    return (f"{lead} Top up to keep going — tap the button below to pay with "
            "TON or USD₮ via TON Connect.")


def precheck(user_id, client=None):
    """Return a top-up prompt to send *instead* of answering, or None to allow.

    Blocks only when the balance can't cover a single interaction. On a backend
    error, allows the turn (unless DP_PAYMENTS_FAIL_CLOSED) so an outage doesn't
    strand users mid-conversation.
    """
    if not enabled():
        return None
    client = client or _client()
    try:
        balance = client.get_balance(user_id)
    except backend.BackendError as e:
        security.audit("payments_balance_error", f"{user_id}: {e}")
        if config.PAYMENTS_FAIL_CLOSED:
            return ("Payments are temporarily unavailable, so I can't continue "
                    "right now. Please try again shortly.")
        return None
    if balance < config.USAGE_COST_USD:
        security.audit("payments_gate_block", f"{user_id}: balance={balance}")
        return pay_prompt(balance)
    return None


def settle(user_id, client=None):
    """Debit one interaction's cost after a successful answer. Best-effort: a
    debit failure is logged, not raised, so it never breaks a delivered reply."""
    if not enabled() or config.USAGE_COST_USD <= 0:
        return
    client = client or _client()
    try:
        client.debit(user_id, config.USAGE_COST_USD)
    except backend.BackendError as e:
        security.audit("payments_debit_error", f"{user_id}: {e}")
