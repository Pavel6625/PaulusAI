"""Business logic: invoices, payment confirmation, balance changes.

Pricing is in USD. An invoice locks the USD price and a TON quote (from a rate
snapshot); the user pays that TON amount or the equivalent USD₮ (1:1). On
confirmation we read what actually arrived on-chain (via PaulusAI), check it
covers the invoice for its currency, and credit the fixed USD price.

Balance mutations run inside a row-locked transaction; the UNIQUE tx_hash is the
final backstop against double-crediting.
"""
import logging
import uuid
from decimal import ROUND_UP, Decimal

from django.conf import settings
from django.db import IntegrityError, transaction

from .models import CURRENCY_TON, CURRENCY_USDT, Account, Invoice, PaymentTransaction
from .paulus_client import PaulusClient, PaulusError

log = logging.getLogger(__name__)

_EPS = Decimal("0.000000001")   # tolerance for on-chain rounding
_NANO = Decimal("0.000000001")
_CENT6 = Decimal("0.000001")    # USD amounts are stored to 6 places

# confirm_invoice outcomes
PAID, PENDING, UNDERPAID = "paid", "pending", "underpaid"


class PaymentError(Exception):
    """A payment/invoice operation could not be completed."""


class InsufficientBalance(PaymentError):
    pass


def get_or_create_account(telegram_id):
    account, _ = Account.objects.get_or_create(telegram_id=telegram_id)
    return account


def _new_memo():
    return f"inv-{uuid.uuid4().hex[:16]}"


def get_ton_usd_rate(client=None):
    """USD price of 1 TON as a Decimal, or None if unavailable (USD₮ still works).

    A configured override wins (handy for tests / a manual peg); otherwise we ask
    PaulusAI, which reads it from the TON API.
    """
    if settings.TON_USD_RATE_OVERRIDE:
        return Decimal(str(settings.TON_USD_RATE_OVERRIDE))
    client = client or PaulusClient()
    try:
        return Decimal(str(client.get_ton_rate("usd")))
    except PaulusError as e:
        log.warning("TON rate unavailable, quoting USD₮ only: %s", e)
        return None


def _ton_quote(amount_usd, rate):
    """TON to charge for *amount_usd* at *rate* (USD/TON), rounded up to nanotons."""
    if not rate or rate <= 0:
        return None
    return (amount_usd / rate).quantize(_NANO, rounding=ROUND_UP)


def create_invoice(telegram_id, amount_usd, client=None):
    """Create a pending invoice priced at *amount_usd*, with a locked TON quote."""
    amount = Decimal(str(amount_usd)).quantize(_CENT6)   # normalise to 6 places
    minimum = Decimal(str(settings.MIN_TOPUP_USD))
    if amount < minimum:
        raise PaymentError(f"amount below minimum top-up of ${minimum}")

    account = get_or_create_account(telegram_id)
    rate = get_ton_usd_rate(client)
    amount_ton = _ton_quote(amount, rate)

    for _ in range(5):   # retry the astronomically unlikely memo collision
        try:
            return Invoice.objects.create(
                account=account, amount_usd=amount, memo=_new_memo(),
                ton_rate=rate, amount_ton=amount_ton)
        except IntegrityError:
            continue
    raise PaymentError("could not allocate a unique invoice memo")


def _covers_invoice(invoice, currency, amount_paid):
    """Whether *amount_paid* in *currency* satisfies the invoice."""
    if currency == CURRENCY_USDT:
        return amount_paid + _EPS >= invoice.amount_usd
    if currency == CURRENCY_TON:
        if invoice.amount_ton is None:      # no TON quote was locked
            return False
        return amount_paid + _EPS >= invoice.amount_ton
    return False


def confirm_invoice(invoice, client=None):
    """Ask PaulusAI whether *invoice* is paid; if adequately paid, record + credit.

    Returns (outcome, account) where outcome is PAID (now or already), PENDING
    (not on-chain yet — retry), or UNDERPAID (found but insufficient). Idempotent.
    """
    if invoice.status == Invoice.PAID:
        return PAID, invoice.account

    client = client or PaulusClient()
    result = client.find_payment(invoice.memo)
    if result is None:
        return PENDING, invoice.account

    currency = result["currency"]
    amount_paid = Decimal(str(result["amount"]))
    if not _covers_invoice(invoice, currency, amount_paid):
        return UNDERPAID, invoice.account

    tx_hash = result["tx_hash"]
    from_address = result.get("from_address", "") or ""
    credited = invoice.amount_usd     # invoice = "buy $X of credit"

    with transaction.atomic():
        account = Account.objects.select_for_update().get(pk=invoice.account_id)
        tx, created = PaymentTransaction.objects.get_or_create(
            tx_hash=tx_hash,
            defaults={
                "invoice": invoice,
                "account": account,
                "currency": currency,
                "amount_paid": amount_paid,
                "amount_usd": credited,
                "from_address": from_address,
            },
        )
        if created:
            account.balance_usd += credited
            account.save(update_fields=["balance_usd"])
            invoice.status = Invoice.PAID
            invoice.save(update_fields=["status"])
            log.info("credited $%s to %s (%s %s, tx %s)", credited,
                     account.telegram_id, amount_paid, currency, tx_hash)
        else:
            account.refresh_from_db()   # already applied (e.g. concurrent confirm)

    return PAID, account


def debit(telegram_id, amount_usd, *, allow_negative=False):
    """Deduct usage cost (USD) from a user's balance. Raises InsufficientBalance
    unless *allow_negative*."""
    amount = Decimal(str(amount_usd))
    with transaction.atomic():
        account = (Account.objects.select_for_update()
                   .get_or_create(telegram_id=telegram_id)[0])
        if not allow_negative and account.balance_usd < amount:
            raise InsufficientBalance("insufficient balance")
        account.balance_usd -= amount
        account.save(update_fields=["balance_usd"])
    return account
