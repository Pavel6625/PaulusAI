# Digital Person — MVP (Phase 1+)

A small but architecturally honest implementation of Phase 1 from the design
doc, now extended with two ideas borrowed from Nous Research's Hermes Agent: a
**procedural skill library** and a **pluggable sandbox backend**. It also
upgrades two MVP stubs into real components: an **OCC appraisal engine** for
emotion and **embeddings-based semantic retrieval** for memory.

This is the skeleton you grow, not a finished product.

## What it does

- Chats from the terminal and remembers you across runs.
- Stores durable facts in long-term memory you can **open and read**, retrieved
  by **semantic similarity** (embeddings), not just keywords.
- Carries an explainable **mood** derived from an **OCC appraisal engine**
  (events → appraisal variables → emotions → a persistent PAD mood).
- Learns **reusable skills** from experience; proposed skills start
  `unverified` and graduate to `verified` once used successfully.
- Runs file/command actions through a **pluggable sandbox** (local / Docker /
  SSH), and **pauses for your explicit approval** on every high-impact action.

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...        # the only place a secret lives
python main.py
```

Chat commands: `/mood`, `/memory`, `/skills`, `/sleep`, `/quit`.

Config via env: `DP_CORE_MODEL` (default `claude-sonnet-4-6`), `DP_SANDBOX`
(`local` | `docker` | `ssh`), `DP_DOCKER_IMAGE`, `DP_SSH_TARGET`. To use a
different LLM provider, edit `llm.py` — the only file that talks to the model.

## How the code maps to the design doc

| Design doc subsystem | File |
|---|---|
| Cognitive core / agentic loop | `agent.py` |
| Memory (episodic + inspectable semantic) | `memory.py` → `memory/` |
| Semantic retrieval (embeddings + vector store) | `vectorstore.py` (Chroma) |
| Consolidation / "subconscious" loop | `agent.sleep()` + `memory.decay()` |
| Affect & personality (OCC appraisal → PAD mood) | `appraisal.py` + `affect.py` |
| Procedural skills / self-improvement (§5.7) | `skills.py` |
| Action & tools | `tools.py` |
| Pluggable sandbox backends (§5.6) | `sandbox.py` |
| Safety gate + trust model (§7) | `security.py` + the gate in `agent.respond()` |
| Model boundary (swappable provider) | `llm.py` |

Your data lives in `memory/`: `episodic.jsonl`, `facts.json`, `semantic.md`
(readable view), `skills.json`, `affect.json`, `audit.log`, and `vectorstore/`
(the derived Chroma index — rebuildable from `facts.json`).

## How the emotion engine works (appraisal.py + affect.py)

An event is described by appraisal *variables* — desirability (vs. goals),
praiseworthiness (vs. standards), who caused it, whether it's actual or
prospective. The OCC engine derives discrete emotions with intensities (joy,
pride, gratitude, fear, shame, ...). Those fold into a persistent **PAD**
(Pleasure-Arousal-Dominance) mood that decays toward a personality baseline.
Everything is legible: the agent can report its mood and the emotion behind it.
This is expressive, explainable emotional behaviour — not a claim of feeling.

## How retrieval works (vectorstore.py)

`facts.json` stays the canonical, inspectable store. Chroma holds a derived
embedding index using a **local** default embedding model (no API key, nothing
leaves the machine). If `chromadb` isn't installed, the code prints a notice and
**falls back to keyword search** so the agent still runs.

## Security posture (what's real in the MVP)

1. **Untrusted data is never instructions.** File contents and command output
   are wrapped in `<untrusted_data>`; the model is told to treat them as data.
   Defence-in-depth; the confirmation gate is the real backstop.
2. **Pluggable sandbox.** File access is confined to `workspace/`; command
   execution runs in the chosen backend (use Docker for untrusted input — it's
   network-disabled with CPU/memory caps).
3. **Human-in-the-loop on high-impact actions.** Writing files, running
   commands, and sending messages each require an explicit `y`. Never generalised.
4. **Self-improvement is gated.** Skills proposed by the consolidation loop are
   `unverified` suggestions; anything they imply still passes the per-action gate.
5. **Secrets isolation.** The API key is read from the environment and never
   enters a prompt, the repo, or memory.
6. **Audit log.** Every action is appended to `memory/audit.log`.

## Honest limitations (and where to grow next)

- **Skill retrieval is keyword-based** (facts use embeddings; skills don't yet).
- **Personality is a 3-trait subset**, not the full Big Five; appraisal templates
  are coarse-grained per event type.
- **`send_message` is simulated**, and the SSH backend leaves workspace sync to
  the operator.
- **Learning is memory + skills + consolidation.** No fine-tuning (that stays
  experimental per the design doc).
- **Prompt-injection defence is partial.** Wrapping + sandbox + gate reduce risk;
  a determined attack on the model is still possible — which is exactly why no
  high-impact action runs without you.

None of this is a sentient being. It is a functional companion — expressive and
consistent by design, not conscious.
