import json
import urllib.error

import pytest

from paulus import billing, config


def test_gate_noop_when_billing_disabled(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "")
    calls = []
    monkeypatch.setattr(billing, "_check_usage", lambda uid: calls.append(uid))

    assert billing.gate("u1") == (True, None)
    assert calls == []


def test_gate_noop_for_cli_user_id_none(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    calls = []
    monkeypatch.setattr(billing, "_check_usage", lambda uid: calls.append(uid))

    assert billing.gate(None) == (True, None)
    assert calls == []


def test_gate_allows_when_service_says_allowed(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(billing, "_check_usage", lambda uid: {"allowed": True, "balance": 5})

    assert billing.gate("u1") == (True, None)


def test_gate_blocks_with_default_message_when_balance_exhausted(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(billing, "_check_usage", lambda uid: {"allowed": False, "balance": 0})

    allowed, message = billing.gate("u1")
    assert allowed is False
    assert message == billing.INSUFFICIENT_BALANCE_MESSAGE


def test_gate_blocks_with_service_provided_message(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(
        billing, "_check_usage",
        lambda uid: {"allowed": False, "balance": 0, "message": "Top up at example.com/billing"},
    )

    allowed, message = billing.gate("u1")
    assert allowed is False
    assert message == "Top up at example.com/billing"


def test_gate_embeds_payment_url_from_response(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(config, "BILLING_TOPUP_URL", "")
    monkeypatch.setattr(billing, "_check_usage", lambda uid: {
        "allowed": False, "balance": 0, "payment_url": "https://pay.example.com/session/abc",
    })

    allowed, message = billing.gate("u1")
    assert allowed is False
    text, url = billing.split_pay_link(message)
    assert text == billing.INSUFFICIENT_BALANCE_MESSAGE
    assert url == "https://pay.example.com/session/abc"


def test_gate_falls_back_to_configured_topup_url_template(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(config, "BILLING_TOPUP_URL", "https://pay.example.com/topup?user={user_id}")
    monkeypatch.setattr(billing, "_check_usage", lambda uid: {"allowed": False, "balance": 0})

    allowed, message = billing.gate("u1")
    assert allowed is False
    text, url = billing.split_pay_link(message)
    assert text == billing.INSUFFICIENT_BALANCE_MESSAGE
    assert url == "https://pay.example.com/topup?user=u1"


def test_gate_response_payment_url_wins_over_configured_template(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(config, "BILLING_TOPUP_URL", "https://pay.example.com/topup?user={user_id}")
    monkeypatch.setattr(billing, "_check_usage", lambda uid: {
        "allowed": False, "balance": 0, "payment_url": "https://pay.example.com/session/xyz",
    })

    _, message = billing.gate("u1")
    _, url = billing.split_pay_link(message)
    assert url == "https://pay.example.com/session/xyz"


def test_gate_no_link_when_neither_source_configured(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(config, "BILLING_TOPUP_URL", "")
    monkeypatch.setattr(billing, "_check_usage", lambda uid: {"allowed": False, "balance": 0})

    _, message = billing.gate("u1")
    text, url = billing.split_pay_link(message)
    assert url is None
    assert text == message  # nothing to strip


def test_gate_service_unavailable_never_carries_a_link(monkeypatch):
    # We don't know the user is actually out of balance here, just that we
    # couldn't ask — so no top-up link should be attached.
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(config, "BILLING_TOPUP_URL", "https://pay.example.com/topup?user={user_id}")

    def _raise(uid):
        raise billing.BillingError("boom")

    monkeypatch.setattr(billing, "_check_usage", _raise)

    _, message = billing.gate("u1")
    text, url = billing.split_pay_link(message)
    assert url is None
    assert text == billing.SERVICE_UNAVAILABLE_MESSAGE


def test_split_pay_link_passes_through_plain_text():
    assert billing.split_pay_link("just a normal reply") == ("just a normal reply", None)


def test_gate_fails_closed_on_service_error(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")

    def _raise(uid):
        raise billing.BillingError("boom")

    monkeypatch.setattr(billing, "_check_usage", _raise)

    allowed, message = billing.gate("u1")
    assert allowed is False
    assert message == billing.SERVICE_UNAVAILABLE_MESSAGE
    assert "billing_check_error" in config.AUDIT_LOG.read_text(encoding="utf-8")


def test_gate_fails_closed_when_response_missing_allowed_key(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(billing, "_check_usage", lambda uid: (_ for _ in ()).throw(
        billing.BillingError("usage check response missing 'allowed': {}")))

    allowed, message = billing.gate("u1")
    assert allowed is False
    assert message == billing.SERVICE_UNAVAILABLE_MESSAGE


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_check_usage_calls_configured_url_with_auth_header(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")
    monkeypatch.setattr(config, "BILLING_CHECK_PATH", "/usage/check")
    monkeypatch.setattr(config, "BILLING_API_KEY", "secret-key")

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({"allowed": True, "balance": 3})

    monkeypatch.setattr("paulus.billing.urllib.request.urlopen", fake_urlopen)

    data = billing._check_usage("u1")

    assert data == {"allowed": True, "balance": 3}
    assert captured["url"] == "https://billing.example.com/usage/check"
    assert captured["headers"]["authorization"] == "Bearer secret-key"
    assert captured["body"] == {"user_id": "u1"}


def test_check_usage_honours_json_body_on_non_2xx_status(monkeypatch):
    """A billing service may signal 'blocked' via HTTP status (e.g. 402) with
    a JSON body, instead of always returning 200. Either convention must work."""
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")

    payload = json.dumps({"allowed": False, "balance": 0, "message": "Top up please"}).encode()

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 402, "Payment Required", None, __import__("io").BytesIO(payload)
        )

    monkeypatch.setattr("paulus.billing.urllib.request.urlopen", fake_urlopen)

    data = billing._check_usage("u1")
    assert data == {"allowed": False, "balance": 0, "message": "Top up please"}


def test_check_usage_raises_on_non_2xx_status_without_usable_body(monkeypatch):
    monkeypatch.setattr(config, "BILLING_API_BASE", "https://billing.example.com")

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", None, __import__("io").BytesIO(b"oops")
        )

    monkeypatch.setattr("paulus.billing.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(billing.BillingError, match="500"):
        billing._check_usage("u1")
