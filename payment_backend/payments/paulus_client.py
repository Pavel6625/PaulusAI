"""Client for the PaulusAI payment validator.

We ask PaulusAI to confirm a payment on-chain. It is the only party that talks to
the TON network; we trust the facts it returns (amount, tx hash) because it reads
them from the chain, not from the user.
"""
import logging

import requests
from django.conf import settings

log = logging.getLogger(__name__)


class PaulusError(Exception):
    """PaulusAI was unreachable or returned an unexpected response."""


class PaulusClient:
    def __init__(self, base_url=None, token=None, timeout=None):
        self.base_url = (base_url or settings.PAULUS_BASE_URL).rstrip("/")
        self.token = token if token is not None else settings.PAULUS_VALIDATOR_TOKEN
        self.timeout = timeout or settings.PAULUS_HTTP_TIMEOUT

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def find_payment(self, memo):
        """Scan our wallet for a payment carrying *memo*.

        Returns the payment dict (tx_hash, currency, amount, from_address, …) when
        found, or None if it isn't on-chain yet (caller may retry). Raises
        PaulusError on transport failure or an unexpected status.
        """
        url = f"{self.base_url}/payments/find"
        try:
            resp = requests.post(url, json={"expected_comment": memo},
                                 headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as e:
            raise PaulusError(f"could not reach PaulusAI: {e}") from e

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None                 # not indexed yet — expected while polling
        raise PaulusError(f"PaulusAI returned HTTP {resp.status_code}: {resp.text}")

    def get_ton_rate(self, currency="usd"):
        """Return the USD (or *currency*) price of 1 TON (float). Raises
        PaulusError; used to quote the TON amount on an invoice."""
        url = f"{self.base_url}/payments/rate"
        try:
            resp = requests.get(url, params={"currency": currency},
                                headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as e:
            raise PaulusError(f"could not reach PaulusAI: {e}") from e
        if resp.status_code != 200:
            raise PaulusError(f"rate lookup failed: HTTP {resp.status_code}")
        return float(resp.json()["rate"])
