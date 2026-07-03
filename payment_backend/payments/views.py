"""HTTP API.

Two audiences, two auth schemes:
  * Mini-app endpoints authenticate the Telegram user from signed initData.
  * Internal endpoints (called by PaulusAI) use a shared bearer token.

Views are thin: parse, authenticate, delegate to services, map outcomes to JSON.
Money is USD; the invoice offers two ways to pay it (TON or USD₮).
"""
import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import services
from .models import Invoice
from .paulus_client import PaulusError
from .telegram import TelegramAuthError, verify_init_data

log = logging.getLogger(__name__)


# --- helpers ----------------------------------------------------------------

def _json_body(request):
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return {}


def _require_telegram_user(request):
    init_data = request.headers.get("X-Telegram-Init-Data")
    if not init_data and request.method == "POST":
        init_data = _json_body(request).get("init_data")
    return verify_init_data(init_data, settings.TELEGRAM_BOT_TOKEN)


def _internal_authorized(request):
    token = settings.INTERNAL_API_TOKEN
    if not token:
        return False  # fail closed: internal API is disabled until a token is set
    return request.headers.get("Authorization", "") == f"Bearer {token}"


def _error(message, status):
    return JsonResponse({"error": message}, status=status)


# --- mini-app endpoints -----------------------------------------------------

@csrf_exempt
@require_GET
def tonconnect_manifest(request):
    return JsonResponse({
        "url": settings.TONCONNECT_APP_URL,
        "name": settings.TONCONNECT_APP_NAME,
        "iconUrl": settings.TONCONNECT_ICON_URL,
    })


@csrf_exempt
@require_POST
def create_invoice_view(request):
    try:
        user = _require_telegram_user(request)
    except TelegramAuthError as e:
        return _error(str(e), 401)

    amount = _json_body(request).get("amount_usd")
    if amount is None:
        return _error("amount_usd is required", 400)
    try:
        invoice = services.create_invoice(user["id"], amount)
    except (services.PaymentError, ArithmeticError, ValueError) as e:
        return _error(str(e), 400)

    # Two ways to pay the same USD price. TON option is present only if a rate
    # was available to quote it; USD₮ is always 1:1 with the price.
    options = {
        "usdt": {
            "jetton_master": settings.USDT_JETTON_MASTER,
            "amount": str(invoice.amount_usd),
            "decimals": 6,
        },
    }
    if invoice.amount_ton is not None:
        options["ton"] = {
            "amount": str(invoice.amount_ton),
            "rate_usd_per_ton": str(invoice.ton_rate),
        }

    return JsonResponse({
        "invoice_id": str(invoice.id),
        "memo": invoice.memo,               # the comment every transfer MUST carry
        "amount_usd": str(invoice.amount_usd),
        "wallet_address": settings.TON_WALLET_ADDRESS,   # recipient for both
        "pay": options,
        "ttl_seconds": settings.INVOICE_TTL_SECONDS,
    }, status=201)


@csrf_exempt
@require_POST
def confirm_invoice_view(request, invoice_id):
    try:
        user = _require_telegram_user(request)
    except TelegramAuthError as e:
        return _error(str(e), 401)

    invoice = get_object_or_404(Invoice, pk=invoice_id)
    if invoice.account.telegram_id != user["id"]:
        return _error("not your invoice", 403)

    try:
        outcome, account = services.confirm_invoice(invoice)
    except PaulusError as e:
        log.warning("validator error confirming %s: %s", invoice.memo, e)
        return _error("payment validator unavailable", 502)

    # paid -> 200, pending -> 202 (poll again), underpaid -> 402.
    http_status = {services.PAID: 200, services.PENDING: 202,
                   services.UNDERPAID: 402}[outcome]
    return JsonResponse({
        "status": outcome,
        "balance_usd": str(account.balance_usd),
    }, status=http_status)


@csrf_exempt
@require_GET
def balance_view(request):
    try:
        user = _require_telegram_user(request)
    except TelegramAuthError as e:
        return _error(str(e), 401)
    account = services.get_or_create_account(user["id"])
    return JsonResponse({"balance_usd": str(account.balance_usd)})


# --- internal endpoints (PaulusAI -> here) ----------------------------------

@csrf_exempt
@require_GET
def internal_balance_view(request, telegram_id):
    if not _internal_authorized(request):
        return _error("unauthorized", 401)
    account = services.get_or_create_account(int(telegram_id))
    return JsonResponse({
        "telegram_id": account.telegram_id,
        "balance_usd": str(account.balance_usd),
    })


@csrf_exempt
@require_POST
def internal_debit_view(request):
    if not _internal_authorized(request):
        return _error("unauthorized", 401)
    body = _json_body(request)
    telegram_id = body.get("telegram_id")
    amount = body.get("amount_usd")
    if telegram_id is None or amount is None:
        return _error("telegram_id and amount_usd are required", 400)
    try:
        account = services.debit(int(telegram_id), amount)
    except services.InsufficientBalance:
        return _error("insufficient balance", 402)
    except (services.PaymentError, ValueError, ArithmeticError) as e:
        return _error(str(e), 400)
    return JsonResponse({
        "telegram_id": account.telegram_id,
        "balance_usd": str(account.balance_usd),
    })
