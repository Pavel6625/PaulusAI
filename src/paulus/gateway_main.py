"""Hermes-style messaging gateway entry point.

  paulus-gateway          # after `pip install`

Environment:
  ANTHROPIC_API_KEY=sk-...                       (or another provider key)
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_ALLOWED_USERS=123456789,987654321     # optional allowlist
"""
import asyncio
import os

from . import config, router, vectorstore
from .gateway.runner import GatewayRunner


async def _main() -> None:
    config.ensure_dirs()
    vectorstore.init()
    router.init()   # no-op unless DP_ROUTING is enabled

    runner = GatewayRunner()

    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        from .gateway.platforms.telegram import TelegramAdapter
        runner.register("telegram", TelegramAdapter(runner))

    if not runner.has_adapters():
        print(
            "No platform adapters configured.\n"
            "Set TELEGRAM_BOT_TOKEN to enable Telegram."
        )
        return

    print("Starting Hermes gateway...")
    await runner.start_all()
    print(f"Adapters:\n{runner.status()}\n")
    print("Running. Press Ctrl+C to stop.")

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nShutting down...")
        await runner.stop_all()
        print("Done.")


def run() -> None:
    """Console-script entry point."""
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
