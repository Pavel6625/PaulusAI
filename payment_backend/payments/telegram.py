"""Verify Telegram Mini App initData.

The mini app sends the `initData` string Telegram signed for it. We recompute the
HMAC exactly as the Telegram spec describes and reject anything that doesn't
match, so a caller can't forge another user's identity.

Spec: https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
"""
import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl


class TelegramAuthError(Exception):
    """initData was missing, malformed, expired, or had a bad signature."""


def verify_init_data(init_data, bot_token, *, max_age_seconds=86400, now=None):
    """Validate *init_data* and return the parsed user dict (id, first_name, …).

    Raises TelegramAuthError on any problem.
    """
    if not init_data:
        raise TelegramAuthError("missing initData")
    if not bot_token:
        raise TelegramAuthError("server is missing TELEGRAM_BOT_TOKEN")

    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = pairs.pop("hash", None)
    if not received_hash:
        raise TelegramAuthError("initData has no hash")

    # Data-check-string: all fields except `hash`, sorted, joined by newlines.
    data_check_string = "\n".join(
        f"{k}={pairs[k]}" for k in sorted(pairs)
    )
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret_key, data_check_string.encode(),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received_hash):
        raise TelegramAuthError("initData signature mismatch")

    # Reject stale initData (replay protection).
    auth_date = int(pairs.get("auth_date", "0") or "0")
    if max_age_seconds and auth_date:
        age = (now if now is not None else time.time()) - auth_date
        if age > max_age_seconds:
            raise TelegramAuthError("initData is expired")

    user_raw = pairs.get("user")
    if not user_raw:
        raise TelegramAuthError("initData has no user")
    try:
        user = json.loads(user_raw)
    except json.JSONDecodeError as e:
        raise TelegramAuthError("initData user is not valid JSON") from e
    if "id" not in user:
        raise TelegramAuthError("initData user has no id")
    return user
