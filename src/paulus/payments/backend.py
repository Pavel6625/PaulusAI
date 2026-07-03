"""Client for the Django payment backend's internal API.

This is the read/spend side of payments (the mirror of the validator, which is
the earn side). The agent uses it to read a user's balance to decide when to
prompt for a top-up, and to debit usage cost as the user interacts.

Django owns the balance; we only ask it questions and request debits.
"""
import logging

from .. import config
from ._http import HTTPError, get_json, post_json

log = logging.getLogger(__name__)


class BackendError(Exception):
    """The payment backend was unreachable or returned an error."""


class InsufficientBalance(BackendError):
    """A debit was refused because the balance is too low."""


class BackendClient:
    def __init__(self, base_url=None, token=None, timeout=None):
        self.base_url = (base_url if base_url is not None
                         else config.PAYMENTS_BACKEND_URL).rstrip("/")
        self.token = (token if token is not None
                      else config.PAYMENTS_BACKEND_TOKEN)
        self.timeout = (timeout if timeout is not None
                        else config.PAYMENTS_HTTP_TIMEOUT)

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_balance(self, telegram_id):
        """Return the user's balance in USD (float). Raises BackendError."""
        url = f"{self.base_url}/internal/accounts/{telegram_id}/balance"
        try:
            data = get_json(url, headers=self._headers(), timeout=self.timeout)
        except HTTPError as e:
            raise BackendError(f"balance lookup failed: {e}") from e
        return float(data.get("balance_usd", 0))

    def debit(self, telegram_id, amount_usd):
        """Deduct *amount_usd* of usage cost; return the new balance (float).

        Raises InsufficientBalance on a 402, BackendError on anything else.
        """
        url = f"{self.base_url}/internal/debit"
        payload = {"telegram_id": telegram_id, "amount_usd": amount_usd}
        try:
            data, _ = post_json(url, payload, headers=self._headers(),
                                timeout=self.timeout)
        except HTTPError as e:
            if e.status == 402:
                raise InsufficientBalance("insufficient balance") from e
            raise BackendError(f"debit failed: {e}") from e
        return float(data.get("balance_usd", 0))
