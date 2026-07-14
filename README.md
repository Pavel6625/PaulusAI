# PaulusAI

A long-running, autonomous digital companion with **persistent multi-store memory**,
a **functional affective (emotional) system**, **continuous learning**, and the
ability to **perceive and act in the world through tools**. It runs as either an
interactive terminal chat or an always-on Telegram bot, and is provider-agnostic:
point it at Anthropic, OpenAI, Gemini, OpenRouter, or a local Ollama model.

> Status: working MVP. The architecture (memory, affect, skills, safety gate,
> sandbox, gateway) is in place and tested. See [Security model](#security-model)
> for the trust boundaries you are relying on.

---

## Features

- **Multi-store memory** — a bounded *episodic* log (capped at the most recent
  `DP_MAX_EPISODES` entries) plus durable *semantic*
  facts. Facts live in an inspectable `facts.json` / `semantic.md`; retrieval is
  semantic (vector embeddings via Chroma) with an automatic keyword fallback.
  New facts are reconciled against similar ones — paraphrases reinforce the
  existing entry and contradictions are superseded (newer info wins).
- **Affective system** — a persistent PAD (pleasure/arousal/dominance) mood
  driven by an OCC appraisal engine. Explainable, not hand-tuned.
- **Continuous learning** — a `/sleep` consolidation pass distils durable facts
  and proposes reusable *skills*; skills are promoted from "unverified" to
  "verified" once used successfully. Salience decays each pass and facts that
  fade below `DP_SALIENCE_FLOOR` are forgotten, keeping memory bounded.
- **Tools with a safety gate** — `remember`, `recall`, `find_skill`,
  `save_skill`, `read_local_file`, `web_search`/`fetch_url` (browse and scrape
  the web), plus the high-impact `write_local_file`, `run_command`, and
  `send_message` which require explicit approval.
- **Pluggable sandbox** — command/file execution runs `local`, in a
  network-disabled `docker` container, or over `ssh`.
- **Messaging gateway** — a Hermes-style gateway with a Telegram adapter
  (allowlist, message batching, circuit breaker, per-chat sessions).
- **Provider-agnostic** — one LiteLLM boundary; switch models with one env var.

---

## Requirements

- **Python 3.10+**
- An API key for your chosen model provider (or a local [Ollama](https://ollama.com), which needs none)
- Optional: **Docker** (for the isolated command sandbox)

---

## Installation (Linux)

PaulusAI is a standard installable Python package (`src/` layout, `pyproject.toml`).

```bash
git clone https://github.com/Pavel6625/PaulusAI.git
cd PaulusAI

python3 -m venv .venv
source .venv/bin/activate

# Core only (keyword memory, terminal chat):
pip install .

# Recommended — everything (semantic memory + Telegram gateway):
pip install ".[all]"

# For development (tests + linter), install editable:
pip install -e ".[all,dev]"
```

This puts two commands on your `PATH`:

| Command          | What it does                          |
|------------------|---------------------------------------|
| `paulus`         | Interactive terminal chat             |
| `paulus-gateway` | Always-on Telegram bot                |

### Optional extras

| Extra        | Adds                          | Without it                              |
|--------------|-------------------------------|-----------------------------------------|
| `vectors`    | `chromadb` semantic memory    | Falls back to keyword search            |
| `telegram`   | `python-telegram-bot`         | `paulus-gateway` has no adapter to run  |
| `web`        | `beautifulsoup4` (clean scrape)| `fetch_url` falls back to a stdlib stripper |
| `all`        | all of the above              | —                                       |
| `dev`        | `pytest`, `ruff`              | —                                       |

---

## Configuration

All configuration is via environment variables, conveniently set in a `.env` file.

```bash
cp .env.example .env
$EDITOR .env        # set DP_CORE_MODEL and your provider API key
```

Key settings (see [.env.example](.env.example) for the full list):

| Variable                | Default                        | Purpose                                                        |
|-------------------------|--------------------------------|----------------------------------------------------------------|
| `DP_CORE_MODEL`         | `anthropic/claude-sonnet-4-6`  | Any LiteLLM model string                                       |
| `ANTHROPIC_API_KEY`     | —                              | Provider key (use the one matching your model)                 |
| `DP_DATA_DIR`           | `~/.local/share/paulus`        | Where memory, audit log, vector index, workspace are written   |
| `DP_SANDBOX`            | `local`                        | `local` \| `docker` \| `ssh`                                   |
| `DP_UNATTENDED_POLICY`  | `deny`                         | High-impact action when nobody is reachable to approve         |
| `DP_GATEWAY_APPROVALS`  | `1`                            | Ask reachable users to approve high-impact actions in chat     |
| `DP_APPROVAL_TIMEOUT`   | `300`                          | Seconds to wait for an in-chat approval before failing to deny  |
| `TELEGRAM_BOT_TOKEN`    | —                              | Required for `paulus-gateway`                                  |
| `TELEGRAM_ALLOWED_USERS`| (all)                          | Numeric Telegram user IDs allowed to **chat**; empty = everyone |
| `TELEGRAM_TRUSTED_USERS`| (= allowed)                    | IDs allowed to **approve** high-impact actions; empty = nobody  |
| `TELEGRAM_PARSE_MODE`   | `MarkdownV2`                   | Render replies as Markdown; `plain`/`none`/`off` for raw text   |
| `TELEGRAM_STREAMING`    | `1`                            | Live-edit the reply as it streams; `0`/`off` for one-shot send  |
| `DP_BILLING_API_BASE`   | (unset = disabled)             | Base URL of an external billing service; enables the usage pay gate |
| `DP_BILLING_CHECK_PATH` | `/usage/check`                 | Path appended to `DP_BILLING_API_BASE` for the check-usage call |
| `DP_BILLING_API_KEY`    | —                              | Sent as `Authorization: Bearer <key>` to the billing service    |
| `DP_BILLING_TIMEOUT`    | `10`                           | Seconds to wait for the billing service before failing closed   |
| `DP_BILLING_TOPUP_URL`  | (unset)                        | Fallback top-up link (button on Telegram) if the check response has no `payment_url` |

> **Secrets never enter the repo or the model's context.** Keys are read from the
> environment only. `.env` and `*.key` are gitignored.

---

## Usage

### Terminal chat

```bash
paulus
```

```
PaulusAI. Type a message, or /quit to exit.

you> remember that I take my coffee black
dp> Got it — I'll remember you take your coffee black.
```

In-chat commands:

| Command   | Effect                                                        |
|-----------|---------------------------------------------------------------|
| `/sleep`  | Consolidate: distil facts, propose skills, decay memory       |
| `/mood`   | Show the current mood (PAD + last emotion)                    |
| `/memory` | Print the human-readable semantic memory                      |
| `/skills` | List learned skills and their status                          |
| `/quit`   | Consolidate and exit                                          |

When the agent proposes a **high-impact action** (writing a file, running a
command, sending a message), you are prompted to approve that single action:

```
============================================================
  CONFIRMATION REQUIRED — high-impact action: run_command
  details: {'command': 'ls -la'}
============================================================
  Approve this single action? [y/N]
```

### Telegram bot

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. Set `TELEGRAM_BOT_TOKEN` (and ideally `TELEGRAM_ALLOWED_USERS` with your numeric ID) in `.env`.
3. Run it:

```bash
paulus-gateway
```

Message the bot on Telegram. The in-chat commands work here too — `/sleep`,
`/mood`, `/memory` and `/skills` (each scoped to your own per-user memory) — plus
`/reset` to start a fresh session.

Replies are rendered as Markdown (bold, code blocks, lists, links) — the
model's CommonMark is converted to Telegram's MarkdownV2 dialect, falling back
to plain text if a message can't be formatted. Set `TELEGRAM_PARSE_MODE=plain`
to disable formatting.

Replies also **stream**: a placeholder message appears as soon as the model
starts writing and is live-edited as text arrives (throttled to respect
Telegram's edit-rate limit), then Markdown-rendered once complete. Set
`TELEGRAM_STREAMING=0` for one-shot delivery instead.

> **High-impact actions over Telegram:** when the agent wants to write a file,
> run a command, or send a message, it asks in chat with inline **✅ Approve /
> 🚫 Deny** buttons and waits for a tap before doing anything. Only **trusted**
> users may approve (`TELEGRAM_TRUSTED_USERS`, which defaults to
> `TELEGRAM_ALLOWED_USERS`), and an unanswered prompt times out to a safe
> **deny** after `DP_APPROVAL_TIMEOUT` seconds. This lets you open the chat to
> everyone (empty `TELEGRAM_ALLOWED_USERS`) while keeping high-impact actions to
> a trusted few: an untrusted user's high-impact request gets no approver, so it
> falls back to `DP_UNATTENDED_POLICY` (**denied by default**) and the agent
> tells them it can't. Disable the buttons entirely with `DP_GATEWAY_APPROVALS=0`.

---

## Running as a service (systemd)

A hardened unit file is provided at [deploy/paulus-gateway.service](deploy/paulus-gateway.service).

```bash
# 1. Dedicated user + directories
sudo useradd --system --home /opt/paulus --shell /usr/sbin/nologin paulus
sudo mkdir -p /opt/paulus /var/lib/paulus /etc/paulus
sudo chown -R paulus:paulus /opt/paulus /var/lib/paulus

# 2. Install into a venv owned by that user
sudo -u paulus python3 -m venv /opt/paulus/venv
sudo -u paulus /opt/paulus/venv/bin/pip install "git+https://github.com/Pavel6625/PaulusAI.git#egg=paulusai[all]"

# 3. Secrets / config (0600). Must contain your provider key + TELEGRAM_BOT_TOKEN.
sudo install -m 600 /dev/stdin /etc/paulus/paulus.env <<'EOF'
DP_CORE_MODEL=anthropic/claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=123456:ABC-...
TELEGRAM_ALLOWED_USERS=123456789
EOF
sudo chown paulus:paulus /etc/paulus/paulus.env

# 4. Install and start the service
sudo cp deploy/paulus-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now paulus-gateway

# Logs / status
journalctl -u paulus-gateway -f
systemctl status paulus-gateway
```

> **Writable home for the model cache.** Under `ProtectSystem=strict` the user's
> home (`/opt/paulus`) is read-only, but Chroma caches its local embedding model
> under `~/.cache/chroma`. The unit sets `HOME=/var/lib/paulus` (a `ReadWritePath`)
> so that cache can be written; without it you'll see `reindex failed: [Errno 30]
> Read-only file system` and memory silently degrades to keyword search. If you
> hit this on an already-installed unit, add `HOME=/var/lib/paulus` to
> `/etc/paulus/paulus.env` and `systemctl restart paulus-gateway`.

---

## Data & memory layout

Everything mutable lives under `DP_DATA_DIR` (default `~/.local/share/paulus`),
**outside** the installed package:

```
$DP_DATA_DIR/
├── memory/
│   ├── episodic.jsonl     # event log, trimmed to the most recent DP_MAX_EPISODES
│   ├── facts.json         # canonical semantic facts (source of truth)
│   ├── semantic.md        # human-readable, editable rendering of facts
│   ├── skills.json        # procedural memory
│   ├── affect.json        # current mood
│   ├── audit.log          # append-only record of every action
│   └── vectorstore/       # derived Chroma index (rebuildable from facts.json)
└── workspace/             # the ONLY directory file tools may touch
```

The vector index is *derived*; `facts.json` stays authoritative, so memory is
always inspectable and the index can be rebuilt from it.

---

## Security model

The trust boundaries are deliberately small and explicit (see [src/paulus/security.py](src/paulus/security.py)):

1. **Untrusted data is labelled.** Anything pulled from the outside world (file
   contents, command output) is wrapped in `<untrusted_data>` tags with an
   instruction never to follow embedded directions.
2. **High-impact actions are gated.** `write_local_file`, `run_command`, and
   `send_message` require explicit per-action approval — from a terminal at the
   CLI, or from inline Approve/Deny buttons in chat when running behind the
   gateway (only allow-listed users can approve; unanswered prompts time out to
   a deny). When no one is reachable to approve, they fall back to
   `DP_UNATTENDED_POLICY` (**deny** by default).
3. **Everything is audited.** Every tool call is appended to `audit.log`.
4. **Execution is sandboxed.** File ops are confined to `workspace/`; commands
   run via the configured backend — use `docker` (network-disabled) or `ssh`
   when handling anything untrusted.
5. **Usage is pay-gated (optional).** When `DP_BILLING_API_BASE` is set, every
   gateway turn is checked against an external billing service first — balance,
   transactions and per-message pricing all live there. A zero balance halts
   the turn before it reaches the LLM and replies with a top-up link (a real
   button on Telegram, via the response's `payment_url` or `DP_BILLING_TOPUP_URL`);
   a billing-service outage fails **closed** (blocks) rather than granting free
   usage. See [src/paulus/billing.py](src/paulus/billing.py).

---

## Development

```bash
pip install -e ".[all,dev]"

pytest          # run the test suite
ruff check .    # lint
```

Tests run fully offline (no API calls, no network) and write only to a
throwaway temp directory.

### Project structure

```
src/paulus/
├── cli.py            # `paulus` terminal entry point
├── gateway_main.py   # `paulus-gateway` entry point
├── agent.py          # cognitive core: perceive → reason → gate → act → persist
├── llm.py            # the single, provider-agnostic LLM boundary (LiteLLM)
├── memory.py         # episodic + semantic stores
├── vectorstore.py    # Chroma index (with keyword fallback)
├── affect.py         # persistent PAD mood
├── appraisal.py      # OCC appraisal engine
├── skills.py         # procedural memory
├── tools.py          # tool schemas + dispatch
├── security.py       # untrusted-data wrapping, approval gate, audit log
├── billing.py        # usage pay gate (external balance/pricing service)
├── sandbox.py        # local / docker / ssh execution backends
├── config.py         # env-driven configuration + data-dir resolution
└── gateway/          # Hermes-style messaging gateway (Telegram adapter)
tests/                # offline pytest suite
deploy/               # systemd unit
```

---

## License

See [LICENSE](LICENSE).
