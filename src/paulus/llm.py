"""The single boundary between the agent and the language model.

Uses LiteLLM for dynamic provider switching. Set DP_CORE_MODEL to any
LiteLLM model string:

  anthropic/claude-sonnet-4-6              (default)
  openai/gpt-4o
  gemini/gemini-1.5-pro
  openrouter/anthropic/claude-sonnet-4-6   (one key reaches many models)
  ollama_chat/llama3                       (no key needed)

The corresponding API key env var must be set (ANTHROPIC_API_KEY,
OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, etc.); LiteLLM reads it
from the environment itself. Ollama needs no key. DP_API_BASE/DP_API_KEY
override the endpoint for the core model only — see config.model_credentials.

Every entry point takes an optional ``model=`` to override the core model for
that call, so a caller can pick a model per turn (e.g. routing) without any
global state; omitting it keeps the configured core model.

agent.py is unaware of the provider — it always sees Anthropic-shaped
response objects. All format conversion lives here.
"""
import json
import sys
from dataclasses import dataclass, field

import litellm

from . import config

litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Normalized response types (Anthropic-shaped, provider-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class _TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class _Response:
    stop_reason: str = "end_turn"
    content: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Format conversion: Anthropic -> OpenAI (LiteLLM's universal wire format)
# ---------------------------------------------------------------------------

def _to_litellm_tools(tool_specs):
    """Anthropic tool spec -> OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tool_specs
    ]


def _to_openai_image(block):
    """Anthropic image block -> OpenAI image_url block (a base64 data URL).

    LiteLLM accepts the OpenAI multimodal shape for every vision-capable
    provider and re-encodes it to each provider's native format, so a single
    data URL works whether the backing model is Claude, GPT-4o, or Gemini.
    """
    src = block["source"]
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"},
    }


def _to_openai_messages(messages):
    """Convert Anthropic-style message history to OpenAI-style.

    Handles four content shapes agent.py produces:
      - plain string  (from _history_to_messages)
      - list of text/tool_use dicts  (assistant turn after tool use)
      - list of tool_result dicts    (user turn returning tool output)
      - list of text/image dicts     (user turn carrying an attached image)
    """
    out = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            texts = [b["text"] for b in content if b.get("type") == "text"]
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    },
                }
                for b in content if b.get("type") == "tool_use"
            ]
            entry = {"role": "assistant", "content": " ".join(texts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)

        elif role == "user":
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            regular = [b for b in content if b.get("type") != "tool_result"]

            for tr in tool_results:
                out.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_use_id"],
                    "content": str(tr.get("content", "")),
                })

            if regular:
                # When the turn carries images, emit OpenAI's multimodal list
                # (text + image_url parts); otherwise collapse to a plain string
                # so text-only history stays byte-identical to before.
                if any(b.get("type") == "image" for b in regular):
                    parts = []
                    for b in regular:
                        if b.get("type") == "image":
                            parts.append(_to_openai_image(b))
                        elif b.get("text"):
                            parts.append({"type": "text", "text": b["text"]})
                    if parts:
                        out.append({"role": "user", "content": parts})
                else:
                    text = " ".join(b.get("text", "") for b in regular)
                    if text:
                        out.append({"role": "user", "content": text})

    return out


def _loads_tool_args(raw):
    """Parse a tool call's JSON arguments, tolerating a provider that appends
    trailing data after a valid object.

    Some models (observed with ollama_chat cloud) occasionally emit a valid
    arguments object followed by stray content (e.g. a second object or prose),
    which makes strict ``json.loads`` raise ``Extra data`` and crash the turn.
    Recover the first complete JSON object and ignore the rest; only a string
    with no leading JSON object at all falls back to ``{}``."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj, _end = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            print(f"[llm] unparseable tool arguments, ignoring: {raw!r}",
                  file=sys.stderr)
            return {}
        print(f"[llm] recovered tool arguments with trailing data: {raw!r}",
              file=sys.stderr)
        return obj


# ---------------------------------------------------------------------------
# Format conversion: LiteLLM response -> Anthropic-shaped _Response
# ---------------------------------------------------------------------------

def _normalize(response):
    choice = response.choices[0]
    msg = choice.message
    finish = choice.finish_reason
    tool_calls = getattr(msg, "tool_calls", None) or []

    # Treat the turn as tool-use whenever the model emitted tool calls, not only
    # when finish_reason says so: some providers report "stop" alongside tool
    # calls, which would otherwise make the caller's loop drop the call.
    stop_reason = "tool_use" if (finish == "tool_calls" or tool_calls) else "end_turn"
    content = []

    if msg.content:
        content.append(_TextBlock(text=msg.content))

    for tc in tool_calls:
        content.append(_ToolUseBlock(
            id=tc.id,
            name=tc.function.name,
            input=_loads_tool_args(tc.function.arguments),
        ))

    return _Response(stop_reason=stop_reason, content=content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def supports_vision(model=None):
    """Whether *model* (default: the core model) can accept image input.

    Used by the gateway to refuse an image up front, but only when the model
    *definitively* can't see. LiteLLM's vision metadata lags new multimodal
    releases, so an unknown/uncatalogued model (flag reported as None, or the
    model not in LiteLLM's map at all) is given the benefit of the doubt and
    attempted — a genuinely blind one then fails at call time and the gateway
    reports it gracefully. Only an explicit ``supports_vision: False`` blocks."""
    try:
        info = litellm.get_model_info(model=model or config.CORE_MODEL)
    except Exception:
        return True                       # not in LiteLLM's map — don't block
    return info.get("supports_vision") is not False


def complete(system, messages, tools=None, model=None):
    """One non-streaming turn. Returns an Anthropic-shaped _Response.

    *model* overrides the configured core model for this call only; its
    credentials are resolved per-model (see config.model_credentials)."""
    model = model or config.CORE_MODEL
    oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
    return _normalize(litellm.completion(
        model=model,
        max_tokens=config.MAX_TOKENS,
        messages=oai_messages,
        tools=_to_litellm_tools(tools) if tools else None,
        **config.model_credentials(model),
    ))


def stream(system, messages, tools=None, on_delta=None, model=None):
    """One streaming turn. Forwards each text delta to ``on_delta(piece)`` as it
    arrives, then returns the same Anthropic-shaped _Response as complete() so
    the caller's tool loop is otherwise unchanged. Tool-call fragments are
    reassembled across chunks before the response is built.

    *model* overrides the configured core model for this call only."""
    model = model or config.CORE_MODEL
    oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
    chunks = litellm.completion(
        model=model,
        max_tokens=config.MAX_TOKENS,
        messages=oai_messages,
        tools=_to_litellm_tools(tools) if tools else None,
        stream=True,
        **config.model_credentials(model),
    )

    text_parts: list[str] = []
    tool_calls: dict[int, dict] = {}   # call index -> {id, name, args}
    finish = None

    for chunk in chunks:
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish = choice.finish_reason
        delta = choice.delta

        piece = getattr(delta, "content", None)
        if piece:
            text_parts.append(piece)
            if on_delta:
                on_delta(piece)

        for tc in getattr(delta, "tool_calls", None) or []:
            slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
            if tc.id:
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn:
                if fn.name:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["args"] += fn.arguments

    # See _normalize: a provider may stream tool-call deltas while reporting a
    # "stop" finish_reason, so trust the accumulated tool calls themselves.
    stop_reason = "tool_use" if (finish == "tool_calls" or tool_calls) else "end_turn"
    content: list = []
    text = "".join(text_parts)
    if text:
        content.append(_TextBlock(text=text))
    for _, slot in sorted(tool_calls.items()):
        content.append(_ToolUseBlock(
            id=slot["id"],
            name=slot["name"],
            input=_loads_tool_args(slot["args"]),
        ))
    return _Response(stop_reason=stop_reason, content=content)
