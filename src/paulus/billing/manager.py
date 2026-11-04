import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BalanceExhausted(Exception):
    """Raised when the user's balance is insufficient for the request."""
    pass

class BillingManager:
    """
    Handles usage reporting and balance checks for PaulusAI.
    
    The manager reads raw cost in USD from the LLM provider (e.g. OpenRouter)
    and reports it to the billing website. The website is responsible for
    applying the 1.2x markup and converting the currency.
    """

    def __init__(self, balance_provider: Any) -> None:
        # balance_provider is an object that can check/update balance via API
        self.balance_provider = balance_provider

    def check_balance(self, user_id: str) -> None:
        """
        Ensures the user has a positive balance before starting a call.
        Blocks the call if balance is <= 0.
        """
        balance = self.balance_provider.get_balance(user_id)
        if balance <= 0:
            raise BalanceExhausted(f"User {user_id} has no remaining balance.")

    def record_usage(self, user_id: str, cost_usd: float) -> Dict[str, Any]:
        """
        Reports the raw USD cost to the billing site.
        Returns the updated balance and total spent.
        """
        if cost_usd < 0:
            logger.warning(f"Received negative cost for user {user_id}: {cost_usd}")
            return {}

        try:
            return self.balance_provider.charge(user_id, cost_usd)
        except Exception as e:
            logger.error(f"Failed to record usage for user {user_id}: {e}")
            # We don't block the agent if the billing site is down, 
            # but we log it as an error.
            return {}
