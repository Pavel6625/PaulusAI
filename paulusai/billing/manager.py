import logging
from typing import Dict, Any
from decimal import Decimal

logger = logging.getLogger(__name__)


class BillingManager:
    """
    Handles usage-based billing calculations for Jev calls.
    Billing is based on the remaining balance.
    """

    def __init__(self, user_id: str, balance_provider: Any) -> None:
        self.user_id = user_id
        self.balance_provider = balance_provider

    async def validate_call_allowed(self) -> bool:
        """Checks if the user has a positive balance to make a call."""
        balance = await self.balance_provider.get_balance(self.user_id)
        return balance > 0

    async def record_usage(self, cost_usd: float) -> Dict[str, Any]:
        """
        Charges the user for a completed call.
        If the call exceeds the remaining balance, the balance is settled to 0.
        Further calls are blocked until top-up.
        """
        try:
            # Raw USD cost reported by agent
            cost_decimal = Decimal(str(cost_usd))

            # Website applies 1.2x multiplier and converts to RUB
            # This logic should typically reside in the billing website,
            # but the core must track the raw USD cost.

            result = await self.balance_provider.charge(self.user_id, cost_decimal)
            return {
                "status": "success",
                "charged_usd": float(cost_decimal),
                "remaining_balance": float(result.get("remaining", 0)),
            }
        except Exception as e:
            logger.error(f"Billing error for user {self.user_id}: {e}")
            return {"status": "error", "message": str(e)}
