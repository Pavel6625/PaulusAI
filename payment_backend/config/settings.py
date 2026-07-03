"""Django settings for the PaulusAI payment backend.

Self-contained and env-driven so it can be lifted into its own repository. All
secrets come from the environment (a local .env is loaded if python-dotenv is
installed). See .env.example for the full list.
"""
import os
from pathlib import Path

try:  # optional dev convenience; production sets real env vars
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_list(name, default=""):
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


# --- Core -------------------------------------------------------------------
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-insecure-change-me")
DEBUG = _env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = _env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
# Comma-separated origins allowed to call the mini-app API from the browser.
CORS_ALLOWED_ORIGINS = _env_list("DJANGO_CORS_ORIGINS", "")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "payments",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "payments.middleware.SimpleCorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# SQLite by default; point DATABASE_URL at Postgres in production if desired.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = []
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Payments ---------------------------------------------------------------
# Telegram bot whose Mini App drives payments; its token also verifies the
# WebApp initData that authenticates each user.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Our TON receiving wallet, in the FRIENDLY (EQ…/UQ…) form the TON Connect
# frontend needs — it's the recipient for BOTH native TON and USD₮ transfers.
# (PaulusAI matches on the raw form.)
TON_WALLET_ADDRESS = os.environ.get("TON_WALLET_ADDRESS", "")
# USD₮ jetton master address (friendly form) the frontend needs to build a USD₮
# transfer. Mainnet USD₮: EQCxE6mUtQJKFnGfaROTKOt1lZbDiiX1kCixRv7Nw2Id_sDs
USDT_JETTON_MASTER = os.environ.get(
    "USDT_JETTON_MASTER", "EQCxE6mUtQJKFnGfaROTKOt1lZbDiiX1kCixRv7Nw2Id_sDs")

# PaulusAI validator (confirms payments on-chain and serves the TON→USD rate).
PAULUS_BASE_URL = os.environ.get("PAULUS_BASE_URL", "http://localhost:8000").rstrip("/")
PAULUS_VALIDATOR_TOKEN = os.environ.get("PAULUS_VALIDATOR_TOKEN", "")
PAULUS_HTTP_TIMEOUT = int(os.environ.get("PAULUS_HTTP_TIMEOUT", "20"))

# Shared secret that PaulusAI must present to our internal balance/debit API.
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")

# Pin the TON price (USD per TON) instead of asking PaulusAI — for tests or a
# manual peg. Empty = fetch the live rate.
TON_USD_RATE_OVERRIDE = os.environ.get("TON_USD_RATE_OVERRIDE", "")

# Invoice rules (USD).
MIN_TOPUP_USD = os.environ.get("MIN_TOPUP_USD", "1")
INVOICE_TTL_SECONDS = int(os.environ.get("INVOICE_TTL_SECONDS", "1800"))

# TON Connect manifest (served at /api/tonconnect-manifest.json).
TONCONNECT_APP_NAME = os.environ.get("TONCONNECT_APP_NAME", "PaulusAI")
TONCONNECT_APP_URL = os.environ.get("TONCONNECT_APP_URL", "https://example.com")
TONCONNECT_ICON_URL = os.environ.get(
    "TONCONNECT_ICON_URL", "https://example.com/icon.png")
