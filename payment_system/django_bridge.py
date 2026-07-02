import requests
import logging

class DjangoPaymentBridge:
    def __init__(self, api_url, token):
        self.api_url = api_url
        self.token = token

    def update_balance(self, telegram_id, amount, tx_hash):
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "telegram_id": telegram_id,
            "amount": amount,
            "tx_hash": tx_hash
        }
        try:
            response = requests.post(f"{self.api_url}/verify/", json=payload, headers=headers)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Django bridge error: {e}")
            return False
