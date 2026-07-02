from flask import Blueprint, request, jsonify
from .core import TONClient, DjangoPaymentBridge
import os

payment_bp = Blueprint('payment', __name__)

# Lazy-loaded clients based on env
def get_ton_client():
    return TONClient(api_key=os.getenv("TON_API_KEY"))

def get_django_bridge():
    return DjangoPaymentBridge(
        api_url=os.getenv("DJANGO_PAYMENT_URL"), 
        token=os.getenv("DJANGO_PAYMENT_TOKEN")
    )

@payment_bp.route('/verify-payment', methods=['POST'])
def verify_payment():
    data = request.json
    user_id = data.get('user_id')
    amount = data.get('amount')
    tx_hash = data.get('tx_hash')

    if not all([user_id, amount, tx_hash]):
        return jsonify({"error": "Missing parameters"}), 400

    # 1. Verify on Blockchain
    ton = get_ton_client()
    if ton.verify_transaction(tx_hash):
        # 2. Update Django Source of Truth
        bridge = get_django_bridge()
        if bridge.update_balance(user_id, amount, tx_hash):
            return jsonify({"status": "success", "message": "Balance updated"}), 200
        return jsonify({"status": "error", "message": "Failed to update Django backend"}), 500
    
    return jsonify({"status": "error", "message": "Invalid transaction hash"}), 400
