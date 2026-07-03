"""Stateless payment validation.

Django owns balances, invoices and idempotency. Our job: given a tx hash or an
invoice memo, prove against the chain whether a real payment (native TON or USD₮)
reached us, and hand back the on-chain facts — currency, amount, sender, memo.
Django decides sufficiency, because only it knows the USD price and the TON quote
it locked into the invoice. We also expose the TON→USD rate used for quoting.

router.py is a thin HTTP shell over these functions.
"""
import logging

from .ton import PaymentVerificationError, TONClient, VerifiedPayment

log = logging.getLogger(__name__)


class PaymentError(Exception):
    """A payment could not be validated. ``.reason`` is a short machine code."""

    def __init__(self, message, reason="rejected"):
        super().__init__(message)
        self.reason = reason


def validate_payment(tx_hash, *, expected_comment=None, ton_client=None):
    """Validate *tx_hash* on-chain and return its VerifiedPayment.

    Raises PaymentError (with a ``.reason``) if the transaction is missing,
    failed, not a payment to us, or stale. ``expected_comment`` optionally binds
    the payment to a specific invoice memo. Amount sufficiency is checked by the
    caller against the invoice, since it depends on the currency.
    """
    if not tx_hash:
        raise PaymentError("tx_hash is required", reason="bad_request")

    ton_client = ton_client or TONClient()
    try:
        verified: VerifiedPayment = ton_client.verify_transaction(tx_hash)
    except PaymentVerificationError as e:
        raise PaymentError(str(e), reason="unverified") from e

    if expected_comment is not None and verified.comment.strip() != expected_comment:
        raise PaymentError(
            "payment memo does not match the expected invoice",
            reason="memo_mismatch")

    log.info("validated %s payment of %g (tx %s)",
             verified.currency, verified.amount, tx_hash)
    return verified


def find_payment(expected_comment, *, ton_client=None):
    """Find a payment by its invoice memo, scanning our wallet's recent txs.

    The usual TON Connect case (no tx hash): Django passes the invoice memo and
    gets back the matching VerifiedPayment (currency, amount, tx_hash) or a
    ``not_found`` PaymentError, which is retryable while a fresh payment is still
    being indexed.
    """
    if not expected_comment:
        raise PaymentError("expected_comment is required", reason="bad_request")

    ton_client = ton_client or TONClient()
    try:
        found = ton_client.find_payment(expected_comment=expected_comment)
    except PaymentVerificationError as e:
        raise PaymentError(str(e), reason="not_found") from e

    log.info("found %s payment of %g for memo %s (tx %s)",
             found.currency, found.amount, expected_comment, found.tx_hash)
    return found


def get_ton_rate(currency="usd", *, ton_client=None):
    """Return the price of 1 TON in *currency* (float). Raises PaymentError."""
    ton_client = ton_client or TONClient()
    try:
        return ton_client.get_rate(currency)
    except PaymentVerificationError as e:
        raise PaymentError(str(e), reason="rate_unavailable") from e
