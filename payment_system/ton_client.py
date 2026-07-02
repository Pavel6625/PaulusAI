import requests
import logging

class TONClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://toncenter.com/api/v2/json"

    def verify_transaction(self, tx_hash, expected_amount, wallet_address):
        payload = {
            "address": wallet_address,
            "limit": 10
        }
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.post(self.base_url + "/getTransactions", json=payload, headers=headers)
            data = response.json()
            if data.get("ok"):
                txs = data.get("result", [])
                for tx in txs:
                    if tx.get("transaction_id") == tx_hash:
                        return True
            return False
        except Exception as e:
            logging.error(f"TON verification error: {e}")
            return False
