"""Data model for the payment backend — the source of truth for balances.

Pricing and balances are in USD. Users pay in native TON or USD₮ (a USD-pegged
jetton); an invoice locks the USD price plus a TON quote (from a rate snapshot)
so the amount to pay is fixed for a short window.

Account   -- a Telegram user and their USD balance.
Invoice   -- a top-up request: a USD price, a UNIQUE memo, and a locked TON
             quote. USD₮ is paid 1:1 with the USD price.
PaymentTransaction -- a confirmed on-chain payment. Its UNIQUE tx_hash is the
             idempotency guard; it records the currency and amount actually paid
             alongside the USD credited.
"""
import uuid
from decimal import Decimal

from django.db import models

# USD amounts: 6 places is plenty for fractional usage costs.
_USD = dict(max_digits=20, decimal_places=6)
# TON has 9 decimal places (1 TON = 1e9 nanotons).
_TON = dict(max_digits=20, decimal_places=9)

CURRENCY_TON = "TON"
CURRENCY_USDT = "USDT"
CURRENCY_CHOICES = [(CURRENCY_TON, CURRENCY_TON), (CURRENCY_USDT, CURRENCY_USDT)]


class Account(models.Model):
    telegram_id = models.BigIntegerField(unique=True, db_index=True)
    balance_usd = models.DecimalField(default=Decimal("0"), **_USD)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Account<{self.telegram_id}: ${self.balance_usd}>"


class Invoice(models.Model):
    PENDING, PAID, EXPIRED = "pending", "paid", "expired"
    STATUS_CHOICES = [(PENDING, PENDING), (PAID, PAID), (EXPIRED, EXPIRED)]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(Account, on_delete=models.CASCADE,
                                related_name="invoices")
    amount_usd = models.DecimalField(**_USD)          # the price / credit value
    memo = models.CharField(max_length=64, unique=True)
    # TON quote locked at creation: pay this many TON, priced at this USD/TON rate.
    ton_rate = models.DecimalField(null=True, blank=True, **_USD)
    amount_ton = models.DecimalField(null=True, blank=True, **_TON)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=PENDING)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Invoice<{self.memo}: ${self.amount_usd} [{self.status}]>"


class PaymentTransaction(models.Model):
    # Unique on-chain hash — the database-level guarantee against double credit.
    tx_hash = models.CharField(max_length=128, unique=True)
    invoice = models.OneToOneField(Invoice, on_delete=models.PROTECT,
                                   related_name="transaction")
    account = models.ForeignKey(Account, on_delete=models.PROTECT,
                                related_name="transactions")
    currency = models.CharField(max_length=8, choices=CURRENCY_CHOICES)
    amount_paid = models.DecimalField(**_TON)         # in the paid currency's unit
    amount_usd = models.DecimalField(**_USD)          # USD credited
    from_address = models.CharField(max_length=128, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return (f"Tx<{self.tx_hash[:12]}…: {self.amount_paid} {self.currency} "
                f"-> ${self.amount_usd}>")
