"""Tests for the TON payment validator (native TON + USD₮ jettons).

The TON API is stubbed; the focus is the trust-critical logic: on-chain
verification (recipient / jetton source / value / staleness), currency
detection, memo matching, the rate feed, and the USD pay-gate.
"""
import time

import pytest

from paulus.payments.service import PaymentError, get_ton_rate, validate_payment
from paulus.payments.service import find_payment as service_find
from paulus.payments.ton import (
    USDT_DECIMALS,
    PaymentVerificationError,
    TONClient,
    VerifiedPayment,
)

WALLET = "0:aaaa"        # our native TON wallet
JWALLET = "0:cccc"       # our USD₮ jetton wallet
NANO = 1_000_000_000


def _ton_tx(*, dest=WALLET, source="0:bbbb", value_ton=1.0, comment="inv-42",
            success=True, utime=None, tx_hash="H"):
    return {
        "hash": tx_hash,
        "success": success,
        "utime": int(utime if utime is not None else time.time()),
        "in_msg": {
            "value": int(value_ton * NANO),
            "source": {"address": source},
            "destination": {"address": dest},
            "decoded_body": {"text": comment},
        },
    }


def _usdt_tx(*, jetton_wallet=JWALLET, dest=WALLET, sender="0:dddd",
             amount_usdt=1.0, comment="inv-42", success=True, utime=None, tx_hash="J"):
    return {
        "hash": tx_hash,
        "success": success,
        "utime": int(utime if utime is not None else time.time()),
        "in_msg": {
            "value": 50_000_000,     # gas nanotons attached to the notification
            "source": {"address": jetton_wallet},
            "destination": {"address": dest},
            "decoded_op_name": "jetton_notify",
            "decoded_body": {
                "amount": str(int(amount_usdt * 10 ** USDT_DECIMALS)),
                "sender": sender,
                # comment lives nested in the forward payload -> tests _find_text
                "forward_payload": {"value": {"value": {"text": comment}}},
            },
        },
    }


def _client(tx):
    c = TONClient(wallet_address=WALLET, usdt_jetton_wallet=JWALLET, max_tx_age=3600)
    c.fetch_transaction = lambda tx_hash: tx
    return c


def _scan_client(txs):
    c = TONClient(wallet_address=WALLET, usdt_jetton_wallet=JWALLET, max_tx_age=3600)
    c.fetch_wallet_transactions = lambda limit: txs
    return c


# --- native TON verification ------------------------------------------------

def test_verify_accepts_ton_payment():
    v = _client(_ton_tx()).verify_transaction("h1")
    assert isinstance(v, VerifiedPayment)
    assert v.currency == "TON" and v.amount == 1.0
    assert v.comment == "inv-42" and v.from_address == "0:bbbb"


def test_verify_rejects_wrong_destination():
    with pytest.raises(PaymentVerificationError):
        _client(_ton_tx(dest="0:9999")).verify_transaction("h1")


def test_verify_matches_destination_case_insensitively():
    assert _client(_ton_tx(dest="0:AAAA")).verify_transaction("h1").amount == 1.0


def test_verify_rejects_failed_tx():
    with pytest.raises(PaymentVerificationError):
        _client(_ton_tx(success=False)).verify_transaction("h1")


def test_verify_rejects_zero_value():
    with pytest.raises(PaymentVerificationError):
        _client(_ton_tx(value_ton=0)).verify_transaction("h1")


def test_verify_rejects_stale_tx():
    with pytest.raises(PaymentVerificationError):
        _client(_ton_tx(utime=1_000_000)).verify_transaction("h1")


def test_fetch_404_maps_to_verification_error(monkeypatch):
    import paulus.payments.ton as ton_mod
    from paulus.payments._http import HTTPError

    def boom(url, **kw):
        raise HTTPError("HTTP 404", status=404)

    monkeypatch.setattr(ton_mod, "get_json", boom)
    with pytest.raises(PaymentVerificationError):
        TONClient(wallet_address=WALLET).verify_transaction("missing")


# --- USD₮ jetton verification -----------------------------------------------

def test_verify_accepts_usdt_payment():
    v = _client(_usdt_tx(amount_usdt=2.5)).verify_transaction("j1")
    assert v.currency == "USDT" and v.amount == 2.5
    assert v.comment == "inv-42" and v.from_address == "0:dddd"


def test_verify_rejects_usdt_from_wrong_jetton_wallet():
    # A notification from some other jetton wallet is NOT our USD₮ — reject.
    with pytest.raises(PaymentVerificationError):
        _client(_usdt_tx(jetton_wallet="0:9999")).verify_transaction("j1")


def test_usdt_ignored_when_not_configured():
    c = TONClient(wallet_address=WALLET, usdt_jetton_wallet="", max_tx_age=3600)
    c.fetch_transaction = lambda h: _usdt_tx()
    with pytest.raises(PaymentVerificationError):
        c.verify_transaction("j1")


# --- validate_payment -------------------------------------------------------

def test_validate_returns_facts():
    v = validate_payment("h", ton_client=_client(_ton_tx(value_ton=2.0)))
    assert v.currency == "TON" and v.amount == 2.0


def test_validate_enforces_expected_comment():
    with pytest.raises(PaymentError) as ei:
        validate_payment("h", ton_client=_client(_ton_tx(comment="inv-99")),
                         expected_comment="inv-42")
    assert ei.value.reason == "memo_mismatch"


def test_validate_maps_verification_failure():
    with pytest.raises(PaymentError) as ei:
        validate_payment("h", ton_client=_client(_ton_tx(dest="0:9999")))
    assert ei.value.reason == "unverified"


def test_validate_requires_tx_hash():
    with pytest.raises(PaymentError) as ei:
        validate_payment("", ton_client=_client(_ton_tx()))
    assert ei.value.reason == "bad_request"


# --- wallet-scan (find_payment) ---------------------------------------------

def test_scan_finds_ton_by_memo():
    txs = [_ton_tx(comment="other", value_ton=5.0),
           _ton_tx(comment="inv-42", value_ton=1.0, tx_hash="WANT")]
    v = _scan_client(txs).find_payment(expected_comment="inv-42")
    assert v.currency == "TON" and v.tx_hash == "WANT"


def test_scan_finds_usdt_by_memo():
    txs = [_usdt_tx(comment="inv-42", amount_usdt=3.0, tx_hash="U")]
    v = _scan_client(txs).find_payment(expected_comment="inv-42")
    assert v.currency == "USDT" and v.amount == 3.0 and v.tx_hash == "U"


def test_scan_returns_newest_match_first():
    txs = [_ton_tx(comment="inv-42", tx_hash="NEW"),
           _ton_tx(comment="inv-42", tx_hash="OLD")]
    assert _scan_client(txs).find_payment(expected_comment="inv-42").tx_hash == "NEW"


def test_scan_skips_wrong_recipient_and_jetton():
    txs = [_ton_tx(dest="0:9999", comment="inv-42"),
           _usdt_tx(jetton_wallet="0:9999", comment="inv-42")]
    with pytest.raises(PaymentVerificationError):
        _scan_client(txs).find_payment(expected_comment="inv-42")


def test_scan_skips_failed_and_stale():
    txs = [_ton_tx(comment="inv-42", success=False),
           _ton_tx(comment="inv-42", utime=1_000_000)]
    with pytest.raises(PaymentVerificationError):
        _scan_client(txs).find_payment(expected_comment="inv-42")


def test_scan_requires_memo():
    with pytest.raises(PaymentVerificationError):
        _scan_client([_ton_tx()]).find_payment(expected_comment="")


def test_service_find_maps_miss_to_not_found():
    with pytest.raises(PaymentError) as ei:
        service_find("inv-42", ton_client=_scan_client([]))
    assert ei.value.reason == "not_found"


def test_service_find_returns_payment():
    v = service_find("inv-42", ton_client=_scan_client([_usdt_tx(comment="inv-42")]))
    assert v.currency == "USDT"


# --- rate feed --------------------------------------------------------------

def test_get_rate_parses_tonapi_shape(monkeypatch):
    import paulus.payments.ton as ton_mod
    monkeypatch.setattr(ton_mod, "get_json",
                        lambda url, **k: {"rates": {"TON": {"prices": {"USD": 4.2}}}})
    assert TONClient(wallet_address=WALLET).get_rate("usd") == 4.2


def test_service_get_ton_rate(monkeypatch):
    import paulus.payments.ton as ton_mod
    monkeypatch.setattr(ton_mod, "get_json",
                        lambda url, **k: {"rates": {"TON": {"prices": {"USD": 3.0}}}})
    assert get_ton_rate("usd") == 3.0


def test_service_rate_error_maps_reason(monkeypatch):
    import paulus.payments.ton as ton_mod
    monkeypatch.setattr(ton_mod, "get_json", lambda url, **k: {"garbage": True})
    with pytest.raises(PaymentError) as ei:
        get_ton_rate("usd")
    assert ei.value.reason == "rate_unavailable"


# --- Flask endpoints (only when the `payments` extra is installed) -----------

@pytest.fixture()
def client(monkeypatch):
    pytest.importorskip("flask")
    from flask import Flask

    from paulus.payments import router, service

    def fake_validate(tx_hash, *, expected_comment=None):
        if tx_hash == "bad":
            raise service.PaymentError("nope", reason="unverified")
        return VerifiedPayment(tx_hash=tx_hash, currency="TON", amount=1.0,
                               from_address="0:bbbb", comment="inv-42",
                               utime=int(time.time()))

    def fake_find(expected_comment):
        if expected_comment == "missing":
            raise service.PaymentError("none yet", reason="not_found")
        return VerifiedPayment(tx_hash="SCANNED", currency="USDT", amount=2.0,
                               from_address="0:dddd", comment=expected_comment,
                               utime=int(time.time()))

    monkeypatch.setattr(router, "validate_payment", fake_validate)
    monkeypatch.setattr(router, "find_payment", fake_find)
    monkeypatch.setattr(router, "get_ton_rate", lambda currency="usd": 4.2)
    app = Flask(__name__)
    app.register_blueprint(router.payment_bp, url_prefix="/payments")
    return app.test_client()


def test_endpoint_validate(client):
    body = client.post("/payments/validate", json={"tx_hash": "good"}).get_json()
    assert body["valid"] and body["currency"] == "TON" and body["amount"] == 1.0


def test_endpoint_validate_rejection(client):
    resp = client.post("/payments/validate", json={"tx_hash": "bad"})
    assert resp.status_code == 402 and resp.get_json()["reason"] == "unverified"


def test_endpoint_find(client):
    body = client.post("/payments/find",
                       json={"expected_comment": "inv-42"}).get_json()
    assert body["valid"] and body["currency"] == "USDT" and body["tx_hash"] == "SCANNED"


def test_endpoint_find_404(client):
    resp = client.post("/payments/find", json={"expected_comment": "missing"})
    assert resp.status_code == 404


def test_endpoint_rate(client):
    body = client.get("/payments/rate?currency=usd").get_json()
    assert body["rate"] == 4.2 and body["token"] == "ton"


def test_endpoint_requires_token(client, monkeypatch):
    from paulus.payments import router
    monkeypatch.setattr(router.config, "PAYMENTS_TOKEN", "s3cret")
    assert client.post("/payments/validate",
                       json={"tx_hash": "good"}).status_code == 401
    assert client.post("/payments/validate", json={"tx_hash": "good"},
                       headers={"Authorization": "Bearer s3cret"}).status_code == 200


# --- payment gate (agent-side balance check + debit) ------------------------

from paulus.payments import backend, gate  # noqa: E402


class _FakeBackend:
    def __init__(self, balance=1.0, fail=False):
        self.balance = balance
        self.fail = fail
        self.debited = None

    def get_balance(self, telegram_id):
        if self.fail:
            raise backend.BackendError("backend down")
        return self.balance

    def debit(self, telegram_id, amount_usd):
        self.debited = (telegram_id, amount_usd)
        self.balance -= amount_usd
        return self.balance


def _enable_gate(monkeypatch, cost=0.02, fail_closed=False):
    monkeypatch.setattr(gate.config, "PAYMENTS_BACKEND_URL", "http://x/api")
    monkeypatch.setattr(gate.config, "PAYMENTS_BACKEND_TOKEN", "tok")
    monkeypatch.setattr(gate.config, "USAGE_COST_USD", cost)
    monkeypatch.setattr(gate.config, "PAYMENTS_FAIL_CLOSED", fail_closed)


def test_gate_inactive_without_config(monkeypatch):
    monkeypatch.setattr(gate.config, "PAYMENTS_BACKEND_URL", "")
    assert gate.enabled() is False
    assert gate.precheck("42", client=_FakeBackend(balance=0.0)) is None


def test_gate_allows_when_funded(monkeypatch):
    _enable_gate(monkeypatch)
    assert gate.precheck("42", client=_FakeBackend(balance=1.0)) is None


def test_gate_blocks_and_prompts_when_low(monkeypatch):
    _enable_gate(monkeypatch)
    prompt = gate.precheck("42", client=_FakeBackend(balance=0.0))
    assert prompt is not None and "top up" in prompt.lower()
    # A positive-but-insufficient balance shows the dollar amount.
    priced = gate.precheck("42", client=_FakeBackend(balance=0.005))
    assert "$0.01" in priced


def test_gate_fails_open_on_backend_error(monkeypatch):
    _enable_gate(monkeypatch, fail_closed=False)
    assert gate.precheck("42", client=_FakeBackend(fail=True)) is None


def test_gate_fails_closed_when_configured(monkeypatch):
    _enable_gate(monkeypatch, fail_closed=True)
    assert gate.precheck("42", client=_FakeBackend(fail=True)) is not None


def test_settle_debits_usage_cost(monkeypatch):
    _enable_gate(monkeypatch, cost=0.03)
    fb = _FakeBackend(balance=1.0)
    gate.settle("42", client=fb)
    assert fb.debited == ("42", 0.03)


def test_settle_noop_when_cost_zero(monkeypatch):
    _enable_gate(monkeypatch, cost=0.0)
    fb = _FakeBackend(balance=1.0)
    gate.settle("42", client=fb)
    assert fb.debited is None


def test_backend_get_balance_parses(monkeypatch):
    monkeypatch.setattr(backend, "get_json", lambda url, **k: {"balance_usd": "1.5"})
    c = backend.BackendClient(base_url="http://x/api", token="tok")
    assert c.get_balance(42) == 1.5


def test_backend_debit_maps_402_to_insufficient(monkeypatch):
    from paulus.payments._http import HTTPError

    def boom(url, payload, **k):
        raise HTTPError("HTTP 402", status=402)

    monkeypatch.setattr(backend, "post_json", boom)
    c = backend.BackendClient(base_url="http://x/api", token="tok")
    with pytest.raises(backend.InsufficientBalance):
        c.debit(42, 0.02)


# --- inspect CLI ------------------------------------------------------------

from argparse import Namespace  # noqa: E402

from paulus.payments import inspect  # noqa: E402


def test_inspect_scan_classifies_usdt(capsys):
    c = _scan_client([_usdt_tx(comment="inv-9", amount_usdt=2.0)])
    inspect.cmd_scan(c, Namespace(limit=5, raw=False))
    out = capsys.readouterr().out
    assert "[OK] USDT 2" in out and "1/1 classified" in out


def test_inspect_tx_reports_ton(capsys):
    inspect.cmd_tx(_client(_ton_tx(value_ton=1.5)), Namespace(tx_hash="h", raw=False))
    assert "[OK] TON 1.5" in capsys.readouterr().out


def test_inspect_memo_miss_returns_nonzero(capsys):
    rc = inspect.cmd_memo(_scan_client([]), Namespace(memo="inv-x"))
    assert rc == 1 and "no match" in capsys.readouterr().out


def test_inspect_help_exits_clean():
    with pytest.raises(SystemExit) as ei:
        inspect.main(["--help"])
    assert ei.value.code == 0
