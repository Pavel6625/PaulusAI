# PaulusAI Payment Backend

Django service that owns user **balances**, **invoices**, and **transactions** for
PaulusAI's crypto payments. Pricing and balances are in **USD**; users pay in
**native TON** or **USD₮** (a USD-pegged jetton). It hosts the Telegram Mini App's
API, confirms payments through the PaulusAI validator, and exposes an internal
API the agent uses to read/debit balances.

> Standalone by design — it lives in its own directory and can be moved to a
> separate repository as-is. It has no import dependency on PaulusAI; the two
> services talk over HTTP.

## Architecture

```
Telegram Mini App (TON Connect)
    │  initData-authenticated
    ▼
THIS SERVICE (Django)  ── owns USD balances / invoices / transactions ──►  SQLite/Postgres
    │  GET  /payments/rate   (quote the TON amount at invoice time)
    │  POST /payments/find    (Bearer PAULUS_VALIDATOR_TOKEN)
    ▼
PaulusAI validator  ── reads TON chain, returns {tx_hash, currency, amount, …}

PaulusAI agent  ── Bearer INTERNAL_API_TOKEN ──►  THIS SERVICE  /api/internal/*
                   (read USD balance for the pay-gate, debit for usage)
```

Trust model: the client is never believed. The Mini App user is authenticated
from Telegram-signed `initData`; the **currency, amount, and tx hash come from the
chain** (via PaulusAI), not the client; and a UNIQUE `tx_hash` makes crediting
idempotent.

## Payment flow

1. `POST /api/invoices` (`{amount_usd}`) → creates a pending invoice with a
   **unique memo**, a locked **TON quote** (from the live rate), and returns both
   pay options: `{memo, amount_usd, wallet_address, pay: {ton, usdt}}`.
2. The Mini App builds a TON Connect transfer (native TON *or* USD₮) to
   `wallet_address` with `memo` as the comment; the user signs.
3. `POST /api/invoices/<id>/confirm` (poll) → we call PaulusAI `/payments/find`
   with the memo. `202 pending` while it's not on-chain yet; `402 underpaid` if
   the amount doesn't cover the invoice for its currency; `200 paid` with the new
   USD balance once satisfied. Idempotent. The credited USD is the fixed invoice
   price (buy $X of credit).

## API

| Method & path | Auth | Purpose |
|---|---|---|
| `GET  /api/tonconnect-manifest.json` | none | TON Connect manifest |
| `POST /api/invoices` | initData | create a top-up invoice |
| `POST /api/invoices/<id>/confirm` | initData | confirm & credit (poll) |
| `GET  /api/balance` | initData | user's balance |
| `GET  /api/internal/accounts/<telegram_id>/balance` | Bearer token | balance (for the agent's pay-gate) |
| `POST /api/internal/debit` | Bearer token | deduct usage cost |

Mini-app endpoints read `initData` from the `X-Telegram-Init-Data` header (or an
`init_data` body field). Internal endpoints require `Authorization: Bearer
$INTERNAL_API_TOKEN`.

## Setup

```bash
cd payment_backend
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in tokens + wallet
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin
python manage.py runserver
python manage.py test           # run the test suite
```

## Configuration

All settings come from the environment — see [.env.example](.env.example). The
two shared secrets must match the PaulusAI side:

- `PAULUS_VALIDATOR_TOKEN` ⇄ PaulusAI's `DP_PAYMENTS_TOKEN`
- `TON_WALLET_ADDRESS` must be the **same wallet** PaulusAI validates against
  (here in friendly `EQ…` form for TON Connect; PaulusAI uses the raw `0:<hex>`
  form for matching).

The default database is SQLite; set up Postgres for production.
