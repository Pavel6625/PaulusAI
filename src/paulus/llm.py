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
from the environment itself. DP_API_BASE/DP_API_KEY
override the endpoint for the core model only — see config.model_credentials.

Every entry point takes an optional ``model=`` to override the core model for
that call, so a caller can pick a model per turn without any global state;
omitting it keeps the configured core model. Passing config.ROUTER_MODEL
hands the choice to an OpenRouter router (see _router_options).

agent.py is unaware of the provider — it always sees Anthropic-shaped
response objects. All format conversion lives here.
"""
import json
import re
import sys
from dataclasses import dataclass, field

import litellm

from . import config
from .billing.manager import BILLING

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


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def loads_json(raw):
    """Parse the JSON value a model *meant* to emit.

    Asked for strict JSON, models reliably do three things anyway: wrap the
    value in a markdown fence, introduce it with prose, and append commentary
    after it. All three make ``json.loads`` raise, so recover from each —
    strip the fence, skip to the first opening brace/bracket, and decode only
    the leading value.

    Raises ValueError when no JSON value can be found at all. Lives here
    because parsing what a model actually said is this module's job, and every
    caller that asks one for JSON hits the same three failures."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON object or array in {_excerpt(raw)}")
    try:
        # raw_decode stops at the end of the first value, ignoring any trailing
        # commentary rather than failing on it.
        value, _end = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError as e:
        raise ValueError(f"unparseable JSON in {_excerpt(raw)}: {e}") from e
    return value


def _excerpt(raw, limit=120):
    text = (raw or "").strip().replace("\n", " ")
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _loads_tool_args(raw):
    """Parse a tool call's JSON arguments, tolerating a provider that wraps or
    pads them.

    Some models (observed with ollama_chat cloud) occasionally emit a valid
    arguments object followed by stray content (e.g. a second object or prose),
    which makes strict ``json.loads`` raise ``Extra data`` and crash the turn.
    Recovery is shared with loads_json; only a string with no JSON value in it
    at all falls back to ``{}``, since a turn is better off with an empty
    argument set than a crash."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)          # the common case: exactly as asked
    except json.JSONDecodeError:
        pass
    try:
        obj = loads_json(raw)
    except ValueError:
        print(f"[llm] unparseable tool arguments, ignoring: {raw!r}",
              file=sys.stderr)
        return {}
    print(f"[llm] recovered malformed tool arguments: {raw!r}", file=sys.stderr)
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


def _router_plugin_id(model):
    """The OpenRouter plugin id that carries *model*'s routing settings, or
    None when *model* isn't a router. ``<author>/auto`` aliases run Jev over
    that author's models; ``openrouter/auto`` is the separate Auto Router."""
    slug = model.removeprefix("openrouter/")
    if slug == "openrouter/auto":
        return "auto-router"
    if slug == "typesafe/jev-router" or slug.endswith("/auto"):
        return "jev-router"
    return None


def _router_options(model):
    """Extra completion kwargs for a call to the configured router: its plugin
    settings in the request body. Empty for any other model, and when no
    setting is given, so the router's account defaults apply."""
    if model != config.ROUTER_MODEL:
        return {}
    plugin_id = _router_plugin_id(model)
    if plugin_id is None:
        return {}
    plugin = {"id": plugin_id}
    if config.ROUTER_COST_TIER:
        plugin["cost_tier"] = config.ROUTER_COST_TIER
    if config.ROUTER_MODELS:
        plugin["allowed_models"] = config.ROUTER_MODELS
    if config.ROUTER_EXCLUDED:
        plugin["excluded_models"] = config.ROUTER_EXCLUDED
    if len(plugin) == 1:
        return {}
    return {"extra_body": {"plugins": [plugin]}}


def _completion(model, **kwargs):
    """litellm.completion on *model*, with its credentials and, for the router,
    its settings.

    A router failure (unreachable, or rejecting its settings) is retried once
    on CORE_MODEL: routing is an optimisation and must never be the reason a
    turn fails. LiteLLM sends a streaming request eagerly, so this covers
    streams too; a stream that breaks after it started is not retried, since
    part of the reply has already been shown."""
    options = _router_options(model)
    try:
        return litellm.completion(model=model, **options, **kwargs)
    except Exception:
        options = _router_options(config.CORE_MODEL)
        return litellm.completion(model=config.CORE_MODEL, **options, **kwargs)


def completion(model=None, user_id=None, **kwargs):
    """The main entry point for LLM completions.
    
    Integrates usage-based billing checks and recording.
    """
    model = model or config.CORE_MODEL
    
    if user_id:
        BILLING.check_balance(user_id)
        
    response = _completion(model, **kwargs)
    
    if user_id:
        cost = getattr(response, 'cost', 0)
        BILLING.record_usage(user_id, cost)
        
    return _normalize(response)


def stream(system, messages, tools=None, on_delta=None, model=None):
    """Stream a turn from the model, forwarding fragments to *on_delta*.

    LiteLLM's streaming protocol is complex: fragments of text and tool
    calls arrive in separate chunks. This function collects them into a
    single final response object, while immediately emitting fragments
    to the caller's callback.

    The returned value is a full normalized _Response object, as if the call
    had been non-streaming.
    """
    model = model or config.CORE_MODEL
    messages_out = _to_openai_messages(messages)
    tools_out = _to_litellm_tools(tools) if tools else None
    
    # Stream is currently not integrated with usage-based billing in the same
    # simple way as completion, but we should ideally check balance here too.
    # For now, we implement the core loop.
    
    stream_resp = litellm.completion(
        model=model,
        messages=messages_out,
        tools=tools_out,
        stream=True,
        **_router_options(model)
    )
    
    full_text = ""
    tool_calls = []
    
    for chunk in stream_resp:
        delta = chunk.choices[0].delta
        if delta.content:
            full_text += delta.content
            if on_delta:
                on_delta(delta.content)
        
        if delta.tool_calls:
            for tc in delta.tool_calls:
                # LiteLLM streams tool calls by index
                idx = tc.index
                if idx >= len(tool_calls):
                    tool_calls.append({"id": tc.id, "name": "", "arguments": ""})
                
                if tc.function.name:
                    tool_calls[idx]["name"] = tc.function.name
                if tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments

    # Construct a mock LiteLLM response for _normalize to process
    class MockResponse:
        def __init__(self, text, tools):
            class Msg:
                def __init__(self, t, ts):
                    self.content = t
                    self.tool_calls = [
                        type('TC', (), {
                            'id': t_id, 
                            'function': type('F', (), {'name': name, 'arguments': args})
                        }) 
                        for t_id, name, args in ts
                    ]
            self.choices = [type('C', (), {
                'message': Msg(text, tools),
                'finish_reason': 'tool_use' if tools else 'stop'
            })]

    # Convert internal tool_calls list to the tuple format MockResponse expects
    tool_tuple = [(tc["id"], tc["name"], tc["arguments"]) for tc in tool_calls]
    return _normalize(MockResponse(full_text, tool_tuple))
