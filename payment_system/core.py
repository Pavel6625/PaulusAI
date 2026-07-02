import requests
import logging

class TONClient:
    """Interface for TON Center API to verify transactions."""
    def __init__(self, api_key, base_url="https://01.tonapi.io/v1/"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def verify_transaction(self, transaction_hash):
        """Checks if a transaction is valid and completed on-chain."""
        url = f"{self.base_url}transactions/{transaction_hash}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # Simplified verification: check if transaction exists and is successful
            return data.get("status") == "confirmed" or "transaction" in data
        except Exception as e:
            logging.error(f"TON verification error: {e}")
            return False

class DjangoPaymentBridge:
    """Bridge to push verified payments to the Django Backend."""
    def __init__(self, api_url, token):
        self.api_url = api_url
        self.token = token

    def update_balance(self, user_id, amount, tx_hash):
        """Securely notifies Django that a payment was verified."""
        payload = {
            "user_id": user_id,
            "amount": amount,
            "transaction_hash": tx_hash,
            "status": "verified"
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Django bridge error: {e}")
            return False
