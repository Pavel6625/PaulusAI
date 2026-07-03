"""Crypto payment validation for PaulusAI (TON Connect).

Topology: a Django-hosted mini app takes the user's TON Connect payment; Django
owns the balance/transaction database. When Django needs to confirm a payment it
calls PaulusAI, which acts as a STATELESS validator — it reads the transaction
from a TON API and returns the on-chain facts (amount, sender, comment). PaulusAI
credits nothing and stores nothing; Django applies the credit and enforces
idempotency.

The trust boundary: the only input is a tx hash. Every fact used to decide a
credit (recipient, amount, memo) is read back from the chain here, never trusted
from the caller.

Two ways in, depending on whether the caller has a tx hash:
  validate_payment(tx_hash, ...)   -- verify a known transaction hash.
  find_payment(expected_comment)   -- scan our wallet for a payment carrying the
                                      invoice memo (the usual TON Connect case,
                                      where only a signed message/BOC exists).

Payments may be native TON or USD₮ jettons; a VerifiedPayment reports which.

Layout:
  ton.py      -- on-chain reads: verify_transaction(), find_payment(), get_rate()
  service.py  -- validate_payment() / find_payment() / get_ton_rate() entry points
  router.py   -- optional Flask blueprint exposing /validate, /find, /rate (needs
                 the `payments` extra), called by Django
"""
from .service import PaymentError, find_payment, get_ton_rate, validate_payment
from .ton import PaymentVerificationError, TONClient, VerifiedPayment

__all__ = [
    "PaymentError",
    "validate_payment",
    "find_payment",
    "get_ton_rate",
    "PaymentVerificationError",
    "TONClient",
    "VerifiedPayment",
]
