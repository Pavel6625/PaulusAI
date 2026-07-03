"""Optional Flask blueprint exposing the validator endpoints (Django -> PaulusAI).

Importing this module requires the `payments` extra (Flask). The logic lives in
service.py, which has no web dependency; this is just auth, request parsing and
status-code mapping.

    from flask import Flask
    from paulus.payments.router import payment_bp
    app = Flask(__name__)
    app.register_blueprint(payment_bp, url_prefix="/payments")

The caller is the Django backend, not an end user, so the endpoints are protected
by a shared bearer token (DP_PAYMENTS_TOKEN).
"""
from flask import Blueprint, jsonify, request

from .. import config
from .service import PaymentError, find_payment, get_ton_rate, validate_payment

payment_bp = Blueprint("payment", __name__)

# Map a PaymentError.reason to an HTTP status.
_STATUS = {
    "bad_request": 400,
    "memo_mismatch": 400,
    "unverified": 402,      # Payment Required — the chain didn't back the claim
    "not_found": 404,       # no matching payment yet (caller may retry)
    "rate_unavailable": 503,
    "rejected": 400,
}


def _payment_json(v):
    return {
        "valid": True,
        "tx_hash": v.tx_hash,
        "currency": v.currency,        # "TON" or "USDT"
        "amount": v.amount,            # in that currency's unit
        "from_address": v.from_address,
        "comment": v.comment,
        "utime": v.utime,
    }


def _error_json(e):
    return jsonify({"valid": False, "reason": e.reason,
                    "message": str(e)}), _STATUS.get(e.reason, 400)


def _authorized(req):
    if not config.PAYMENTS_TOKEN:
        return True   # auth disabled (dev); see config note
    return req.headers.get("Authorization", "") == f"Bearer {config.PAYMENTS_TOKEN}"


def _unauthorized():
    return jsonify({"valid": False, "reason": "unauthorized",
                    "message": "missing or invalid token"}), 401


@payment_bp.route("/validate", methods=["POST"])
def validate():
    if not _authorized(request):
        return _unauthorized()
    data = request.get_json(silent=True) or {}
    try:
        v = validate_payment(data.get("tx_hash"),
                             expected_comment=data.get("expected_comment"))
    except PaymentError as e:
        return _error_json(e)
    return jsonify(_payment_json(v)), 200


@payment_bp.route("/find", methods=["POST"])
def find():
    """Locate a payment by invoice memo (no tx hash needed)."""
    if not _authorized(request):
        return _unauthorized()
    data = request.get_json(silent=True) or {}
    try:
        v = find_payment(data.get("expected_comment"))
    except PaymentError as e:
        return _error_json(e)
    return jsonify(_payment_json(v)), 200


@payment_bp.route("/rate", methods=["GET"])
def rate():
    """TON price in a fiat currency (default USD), used to quote invoices."""
    if not _authorized(request):
        return _unauthorized()
    currency = request.args.get("currency", "usd")
    try:
        value = get_ton_rate(currency)
    except PaymentError as e:
        return _error_json(e)
    return jsonify({"token": "ton", "currency": currency, "rate": value}), 200
