"""Tests for the payment backend.

The PaulusAI validator is stubbed (a fake client), so nothing here touches the
network. Focus: USD pricing with a locked TON quote, currency-aware confirmation
(TON and USD₮) with crediting + idempotency, underpayment, balance/debit,
initData auth, and the internal token gate.
"""
from decimal import Decimal

from django.test import Client, TestCase, override_settings

from .models import Account, Invoice, PaymentTransaction
from .services import (
    PAID,
    PENDING,
    UNDERPAID,
    InsufficientBalance,
    PaymentError,
    confirm_invoice,
    create_invoice,
    debit,
)


class FakeClient:
    """Stand-in for PaulusClient: a fixed rate and a canned find result."""

    def __init__(self, result=None, rate=2.0):
        self.result = result
        self.rate = rate

    def get_ton_rate(self, currency="usd"):
        return self.rate

    def find_payment(self, memo):
        return self.result


def _payment(currency, amount, tx_hash="TX1", from_address="0:bbbb"):
    return {"tx_hash": tx_hash, "currency": currency, "amount": float(amount),
            "from_address": from_address}


@override_settings(TON_USD_RATE_OVERRIDE="", MIN_TOPUP_USD="1")
class InvoiceTests(TestCase):
    def test_create_locks_ton_quote_from_rate(self):
        # $2 at 2 USD/TON -> 1 TON quote.
        inv = create_invoice(42, "2", client=FakeClient(rate=2.0))
        self.assertEqual(inv.amount_usd, Decimal("2"))
        self.assertEqual(inv.ton_rate, Decimal("2"))
        self.assertEqual(inv.amount_ton, Decimal("1.000000000"))
        self.assertTrue(inv.memo.startswith("inv-"))

    def test_create_without_rate_still_offers_usdt(self):
        # Rate unavailable -> no TON quote, but the invoice (USD₮) still works.
        class NoRate(FakeClient):
            def get_ton_rate(self, currency="usd"):
                from .paulus_client import PaulusError
                raise PaulusError("down")

        inv = create_invoice(42, "2", client=NoRate())
        self.assertIsNone(inv.amount_ton)

    def test_below_minimum_rejected(self):
        with self.assertRaises(PaymentError):
            create_invoice(42, "0.5", client=FakeClient(rate=2.0))


@override_settings(TON_USD_RATE_OVERRIDE="2")
class ConfirmTests(TestCase):
    def _invoice(self):
        return create_invoice(42, "2", client=FakeClient(rate=2.0))

    def test_pending_when_not_onchain(self):
        inv = self._invoice()
        outcome, account = confirm_invoice(inv, client=FakeClient(result=None))
        self.assertEqual(outcome, PENDING)
        self.assertEqual(account.balance_usd, Decimal("0"))

    def test_credit_usdt_payment(self):
        inv = self._invoice()
        outcome, account = confirm_invoice(
            inv, client=FakeClient(result=_payment("USDT", "2")))
        self.assertEqual(outcome, PAID)
        self.assertEqual(account.balance_usd, Decimal("2"))
        tx = PaymentTransaction.objects.get()
        self.assertEqual(tx.currency, "USDT")

    def test_credit_ton_payment(self):
        inv = self._invoice()   # quote is 1 TON
        outcome, account = confirm_invoice(
            inv, client=FakeClient(result=_payment("TON", "1")))
        self.assertEqual(outcome, PAID)
        # Credited the fixed USD price, not the TON amount.
        self.assertEqual(account.balance_usd, Decimal("2"))

    def test_underpaid_ton_not_credited(self):
        inv = self._invoice()   # needs 1 TON
        outcome, account = confirm_invoice(
            inv, client=FakeClient(result=_payment("TON", "0.4")))
        self.assertEqual(outcome, UNDERPAID)
        self.assertEqual(account.balance_usd, Decimal("0"))
        self.assertEqual(PaymentTransaction.objects.count(), 0)

    def test_underpaid_usdt_not_credited(self):
        inv = self._invoice()
        outcome, _ = confirm_invoice(
            inv, client=FakeClient(result=_payment("USDT", "1.5")))
        self.assertEqual(outcome, UNDERPAID)

    def test_idempotent(self):
        inv = self._invoice()
        client = FakeClient(result=_payment("USDT", "2"))
        confirm_invoice(inv, client=client)
        inv.refresh_from_db()
        outcome, account = confirm_invoice(inv, client=client)
        self.assertEqual(outcome, PAID)
        self.assertEqual(account.balance_usd, Decimal("2"))
        self.assertEqual(PaymentTransaction.objects.count(), 1)


class BalanceTests(TestCase):
    def test_debit_reduces_balance(self):
        Account.objects.create(telegram_id=7, balance_usd=Decimal("2.0"))
        self.assertEqual(debit(7, "0.5").balance_usd, Decimal("1.5"))

    def test_debit_insufficient_raises(self):
        Account.objects.create(telegram_id=7, balance_usd=Decimal("0.1"))
        with self.assertRaises(InsufficientBalance):
            debit(7, "0.5")


@override_settings(INTERNAL_API_TOKEN="s3cret", TELEGRAM_BOT_TOKEN="botT",
                   TON_USD_RATE_OVERRIDE="2")
class ApiTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_invoice_requires_valid_init_data(self):
        resp = self.client.post("/api/invoices",
                                data={"amount_usd": "2", "init_data": "forged"},
                                content_type="application/json")
        self.assertEqual(resp.status_code, 401)

    def test_internal_balance_requires_token(self):
        Account.objects.create(telegram_id=7, balance_usd=Decimal("3.0"))
        self.assertEqual(
            self.client.get("/api/internal/accounts/7/balance").status_code, 401)
        ok = self.client.get("/api/internal/accounts/7/balance",
                             HTTP_AUTHORIZATION="Bearer s3cret")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["balance_usd"], "3.000000")

    def test_internal_debit(self):
        Account.objects.create(telegram_id=7, balance_usd=Decimal("2.0"))
        resp = self.client.post("/api/internal/debit",
                                data={"telegram_id": 7, "amount_usd": "0.5"},
                                content_type="application/json",
                                HTTP_AUTHORIZATION="Bearer s3cret")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["balance_usd"], "1.500000")

    def test_internal_debit_insufficient(self):
        Account.objects.create(telegram_id=7, balance_usd=Decimal("0.1"))
        resp = self.client.post("/api/internal/debit",
                                data={"telegram_id": 7, "amount_usd": "0.5"},
                                content_type="application/json",
                                HTTP_AUTHORIZATION="Bearer s3cret")
        self.assertEqual(resp.status_code, 402)


class TelegramAuthTests(TestCase):
    def test_round_trip_signature(self):
        import hashlib
        import hmac
        import json
        from urllib.parse import urlencode

        from .telegram import verify_init_data

        bot_token = "12345:TESTTOKEN"
        user = {"id": 42, "first_name": "Ada"}
        fields = {"auth_date": "9999999999", "user": json.dumps(user)}
        dcs = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
        secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        fields["hash"] = hmac.new(secret, dcs.encode(), hashlib.sha256).hexdigest()

        self.assertEqual(verify_init_data(urlencode(fields), bot_token)["id"], 42)
