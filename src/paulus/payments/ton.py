"""On-chain verification of TON payments — native TON and USD₮ jettons.

Two entry points, both returning a VerifiedPayment (or raising
PaymentVerificationError):

  verify_transaction(tx_hash)      -- when the caller already has a tx hash.
  find_payment(expected_comment)   -- scan our wallet's recent inbound payments
                                      for one carrying a memo.

A payment is either native TON (an inbound transfer to our wallet) or a USD₮
jetton transfer (which arrives as a jetton *transfer notification* from OUR USD₮
jetton wallet). We report which currency was used and the amount in that
currency's own units; the caller (Django) decides sufficiency against the
invoice, since only it knows the USD price and the locked TON quote.

We never trust an amount/recipient the caller claims — all of it is read here.
Also exposes get_rate() for the TON→USD price used to quote invoices.

NOTE: the jetton-notification field shapes below follow tonapi's decoded output.
They are covered by unit tests with representative payloads, but should be
sanity-checked against a real USD₮ payment before production.
"""
import time
import urllib.parse
from dataclasses import dataclass

from .. import config
from ._http import HTTPError, get_json

NANO_PER_TON = 1_000_000_000
USDT_DECIMALS = 6

CURRENCY_TON = "TON"
CURRENCY_USDT = "USDT"

# tonapi op names that denote an inbound jetton transfer notification.
_JETTON_NOTIFY_OPS = {"jetton_notify", "jetton_notification", "transfer_notification"}


class PaymentVerificationError(Exception):
    """The transaction could not be verified as a valid payment to us."""


@dataclass(frozen=True)
class VerifiedPayment:
    """The trustworthy, on-chain facts about a verified payment."""

    tx_hash: str
    currency: str              # "TON" or "USDT"
    amount: float              # in the currency's own unit (TON, or USD₮ dollars)
    from_address: str
    comment: str               # the transfer's memo, "" if none
    utime: int                 # on-chain timestamp (unix seconds)


def _account_id(address):
    """Reduce a TON address to a comparable key (lowercased hex account id)."""
    if not address:
        return ""
    address = address.strip()
    if ":" in address:
        return address.split(":", 1)[1].lower()
    return address.lower()


def _addresses_match(a, b):
    return bool(_account_id(a)) and _account_id(a) == _account_id(b)


def _addr(value):
    """Normalise an address that tonapi may give as a string or {"address": …}."""
    if isinstance(value, dict):
        return value.get("address", "")
    return value or ""


def _text_comment(in_msg):
    """Extract a native transfer's text comment (tonapi decodes it at
    decoded_body.text; a couple of sibling shapes are checked defensively)."""
    if not isinstance(in_msg, dict):
        return ""
    body = in_msg.get("decoded_body")
    if isinstance(body, dict) and isinstance(body.get("text"), str):
        return body["text"]
    for key in ("comment", "message"):
        if isinstance(in_msg.get(key), str):
            return in_msg[key]
    return ""


def _find_text(obj, depth=0):
    """Recursively find the first {"text": <str>} — jetton comments live inside a
    nested forward_payload whose exact shape varies, so we search rather than
    hard-code the path."""
    if depth > 6:
        return ""
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            return obj["text"]
        for v in obj.values():
            found = _find_text(v, depth + 1)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_text(v, depth + 1)
            if found:
                return found
    return ""


def _positive_amount(raw, decimals):
    """Return raw/10**decimals as a float if it is a positive integer, else None."""
    if not isinstance(raw, (int, str)):
        return None
    try:
        units = int(raw)
    except (TypeError, ValueError):
        return None
    return units / (10 ** decimals) if units > 0 else None


class TONClient:
    """Reads and validates payments against a TON API (tonapi.io v2)."""

    def __init__(self, api_base=None, api_key=None, wallet_address=None,
                 usdt_jetton_wallet=None, max_tx_age=None, timeout=None,
                 scan_limit=None):
        self.api_base = (api_base or config.TON_API_BASE).rstrip("/")
        self.api_key = api_key if api_key is not None else config.TON_API_KEY
        self.wallet_address = (wallet_address if wallet_address is not None
                               else config.TON_WALLET_ADDRESS)
        self.usdt_jetton_wallet = (usdt_jetton_wallet if usdt_jetton_wallet is not None
                                   else config.USDT_JETTON_WALLET)
        self.max_tx_age = (max_tx_age if max_tx_age is not None
                           else config.TON_MAX_TX_AGE)
        self.timeout = timeout if timeout is not None else config.TON_HTTP_TIMEOUT
        self.scan_limit = (scan_limit if scan_limit is not None
                           else config.TON_SCAN_LIMIT)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _is_stale(self, utime, now):
        if not self.max_tx_age or not utime:
            return False
        return (now - utime) > self.max_tx_age

    # --- network ------------------------------------------------------------

    def fetch_transaction(self, tx_hash):
        url = f"{self.api_base}/blockchain/transactions/{tx_hash}"
        try:
            return get_json(url, headers=self._headers(), timeout=self.timeout)
        except HTTPError as e:
            if e.status == 404:
                raise PaymentVerificationError("transaction not found on-chain") from e
            raise PaymentVerificationError(f"could not reach TON API: {e}") from e

    def fetch_wallet_transactions(self, limit):
        addr = urllib.parse.quote(self.wallet_address, safe="")
        url = (f"{self.api_base}/blockchain/accounts/{addr}"
               f"/transactions?limit={int(limit)}")
        try:
            data = get_json(url, headers=self._headers(), timeout=self.timeout)
        except HTTPError as e:
            raise PaymentVerificationError(f"could not reach TON API: {e}") from e
        txs = data.get("transactions")
        return txs if isinstance(txs, list) else []

    def get_rate(self, currency="usd"):
        """Return the price of 1 TON in *currency* (float). Raises on failure."""
        url = f"{self.api_base}/rates?tokens=ton&currencies={currency}"
        try:
            data = get_json(url, headers=self._headers(), timeout=self.timeout)
        except HTTPError as e:
            raise PaymentVerificationError(f"could not fetch TON rate: {e}") from e
        try:
            return float(data["rates"]["TON"]["prices"][currency.upper()])
        except (KeyError, TypeError, ValueError) as e:
            raise PaymentVerificationError("unexpected rate response shape") from e

    # --- classification -----------------------------------------------------

    def _classify(self, tx, tx_hash):
        """Turn a transaction into a VerifiedPayment, or None if it isn't a
        payment to us. Recipient/jetton-source checks live here; success and
        staleness are enforced by the callers."""
        in_msg = tx.get("in_msg") or {}
        op = (in_msg.get("decoded_op_name") or "").lower()
        utime = int(tx.get("utime", 0))

        if op in _JETTON_NOTIFY_OPS:
            # A USD₮ payment reaches us as a notification FROM our own USD₮ jetton
            # wallet — that source check is what proves it's the right jetton and
            # addressed to us.
            if not self.usdt_jetton_wallet:
                return None
            if not _addresses_match(_addr(in_msg.get("source")),
                                    self.usdt_jetton_wallet):
                return None
            body = in_msg.get("decoded_body") or {}
            amount = _positive_amount(body.get("amount"), USDT_DECIMALS)
            if amount is None:
                return None
            return VerifiedPayment(
                tx_hash=tx_hash, currency=CURRENCY_USDT, amount=amount,
                from_address=_addr(body.get("sender")),
                comment=_find_text(body), utime=utime)

        # Native TON transfer to our wallet.
        source = _addr(in_msg.get("source"))
        if not source or not self.wallet_address:
            return None
        if not _addresses_match(_addr(in_msg.get("destination")), self.wallet_address):
            return None
        amount = _positive_amount(in_msg.get("value"), 9)   # nanotons -> TON
        if amount is None:
            return None
        return VerifiedPayment(
            tx_hash=tx_hash, currency=CURRENCY_TON, amount=amount,
            from_address=source, comment=_text_comment(in_msg), utime=utime)

    # --- verification -------------------------------------------------------

    def verify_transaction(self, tx_hash, *, now=None):
        """Verify *tx_hash* is a successful payment (TON or USD₮) to us."""
        tx = self.fetch_transaction(tx_hash)
        if tx.get("success") is False or tx.get("aborted") is True:
            raise PaymentVerificationError("transaction did not succeed on-chain")

        payment = self._classify(tx, tx_hash)
        if payment is None:
            raise PaymentVerificationError(
                "transaction is not a recognised payment to our wallet")
        now = now if now is not None else time.time()
        if self._is_stale(payment.utime, now):
            raise PaymentVerificationError("transaction is too old")
        return payment

    def find_payment(self, *, expected_comment, limit=None, now=None):
        """Scan recent inbound payments for one carrying *expected_comment*.

        Returns the newest qualifying VerifiedPayment (valid recipient/jetton
        source, positive amount, not stale, memo matches). Raises
        PaymentVerificationError if none is found — a miss is expected while a
        fresh payment is still being indexed, so callers may retry. Amount
        sufficiency is the caller's call (it depends on the currency + invoice).
        """
        if not expected_comment:
            raise PaymentVerificationError(
                "a memo to match is required for a wallet scan")
        if not self.wallet_address:
            raise PaymentVerificationError(
                "no receiving wallet configured (DP_TON_WALLET_ADDRESS)")

        now = now if now is not None else time.time()
        limit = limit if limit is not None else self.scan_limit
        for tx in self.fetch_wallet_transactions(limit):
            if tx.get("success") is False or tx.get("aborted") is True:
                continue
            payment = self._classify(tx, tx.get("hash", ""))
            if payment is None:
                continue
            if payment.comment.strip() != expected_comment:
                continue
            if self._is_stale(payment.utime, now):
                continue
            return payment

        raise PaymentVerificationError(
            "no matching payment found in recent wallet transactions")
