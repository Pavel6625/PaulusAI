import logging
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

class BalanceProvider(Protocol):
    def get_balance(self, user_id: str) -> float: ...
    def update_balance(self, user_id: str, amount: float) -> bool: ...

class BalanceExhausted(Exception):
    def __init__(self, user_id: str, balance: float):
        self.user_id = user_id
        self.balance = balance
        super().__init__(f"User {user_id} has exhausted their balance (current: {balance:.4f})")

class BillingManager:
    def __init__(self, balance_provider: Optional[BalanceProvider] = None):
        self.balance_provider = balance_provider

    def check_balance(self, user_id: str):
        if not self.balance_provider:
            return # Assume unlimited if no provider is configured (dev mode)
        
        balance = self.balance_provider.get_balance(user_id)
        if balance <= 0:
            raise BalanceExhausted(user_id, balance)

    def record_usage(self, user_id: str, cost: float):
        if not self.balance_provider:
            logger.info(f"[Billing] Dev mode: User {user_id} spent ${cost:.6f}")
            return

        if cost <= 0:
            return

        # Record the deduction. The balance_provider implementation (via API)
        # should handle the 1.2x markup if that's part of the backend logic.
        success = self.balance_provider.update_balance(user_id, -cost)
        if not success:
            logger.error(f"[Billing] Failed to update balance for user {user_id} for cost ${cost:.6f}")

# Singleton for the agent process
BILLING = BillingManager()
