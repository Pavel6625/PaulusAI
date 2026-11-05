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
that call, so a caller can pick a model per turn without any global state;
omitting it keeps the configured core model. Passing config.ROUTER_MODEL
hands the choice to an OpenRouter router (see _router_options).

agent.py is unaware of the provider — it always sees Anthropic-shaped
response objects. All format conversion lives here.

Inside a metered turn (billing.metered), every call's token use and cost is
handed to billing — for a call through OpenRouter, the cost OpenRouter itself
records for that generation (its /generation data, read in the background);
otherwise the cost in the response, else LiteLLM's price-map estimate — and a
call is refused with billing.BalanceExhausted once the turn's meter has been
stopped.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import litellm

from . import billing, config

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
    part of the reply has already been shown.

    Returns ``(response, model)`` — the model that actually served the call,
    which billing needs to price it. Raises billing.BalanceExhausted, before
    sending anything, when the current metered turn may spend no more."""
    billing.ensure_can_spend()
    try:
        return litellm.completion(model=model, **kwargs,
                                  **config.model_credentials(model),
                                  **_router_options(model)), model
    except Exception as e:
        if model != config.ROUTER_MODEL or model == config.CORE_MODEL:
            raise
        print(f"[llm] router {model} failed, falling back to "
              f"{config.CORE_MODEL}: {e}", file=sys.stderr)
        return litellm.completion(model=config.CORE_MODEL, **kwargs,
                                  **config.model_credentials(config.CORE_MODEL)), \
            config.CORE_MODEL


# ---------------------------------------------------------------------------
# Usage & cost (for billing)
# ---------------------------------------------------------------------------

# Where LiteLLM's OpenRouter provider stashes the body's usage.cost.
_HIDDEN_COST_KEY = "llm_provider-x-litellm-response-cost"


def _field(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_cost(value):
    """A cost as a float, accepting a bare number or a ``{"total_cost": n}``
    breakdown; None for anything else (bools included)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return _as_cost(value.get("total_cost"))
    return None


def _provider_cost(usage, hidden_params=None):
    """The USD cost the provider itself reported for a call, or None.

    OpenRouter returns it on every response as ``usage.cost`` (no request flag
    needed; LiteLLM keeps it on Usage and, for non-streamed calls, also in
    _hidden_params). With BYOK, ``cost`` is only OpenRouter's fee and the
    provider's charge is ``cost_details.upstream_inference_cost``, so the two
    are added."""
    headers = _field(hidden_params, "additional_headers") or {}
    candidates = (
        _as_cost(_field(headers, _HIDDEN_COST_KEY)),
        _as_cost(_field(usage, "cost")),
        _as_cost(_field(hidden_params, "response_cost")),
    )
    # LiteLLM may initialize Usage.cost to 0 even when OpenRouter's raw cost
    # was retained separately in hidden params. Prefer any non-zero reported
    # value before accepting an explicit zero and consulting /generation.
    cost = next((value for value in candidates if value not in (None, 0.0)), None)
    if cost is None:
        cost = next((value for value in candidates if value is not None), None)
    if cost is not None and _field(usage, "is_byok"):
        upstream = _as_cost(_field(_field(usage, "cost_details"),
                                   "upstream_inference_cost"))
        cost += upstream or 0.0
    return cost


_OPENROUTER_HOST = "openrouter.ai"
# Delays (seconds) between attempts to read a generation record: OpenRouter
# only has it a moment after the reply finishes, and 404s until then.
_GENERATION_RETRY_DELAYS = (0.5, 1, 2, 3, 5, 8)


def _openrouter_key(model):
    """The OpenRouter API key *model* is called with, or None when the call
    doesn't go to OpenRouter. Covers ``openrouter/...`` models and a core model
    pointed at OpenRouter through DP_API_BASE (the key is then DP_API_KEY)."""
    if model.startswith("openrouter/"):
        return config.OPENROUTER_API_KEY or None
    creds = config.model_credentials(model)
    if _OPENROUTER_HOST in (creds.get("api_base") or ""):
        return creds.get("api_key") or config.OPENROUTER_API_KEY or None
    return None


def _uses_openrouter(model):
    if model.startswith("openrouter/"):
        return True
    return _OPENROUTER_HOST in (config.model_credentials(model).get("api_base") or "")


def _openrouter_generation_cost(request_id, api_key):
    """OpenRouter's own USD charge for generation *request_id* — what its
    Generation Data shows as ``usage`` (``total_cost`` is the same figure).

    Retries while the record isn't there yet (404) or the request fails, for
    up to ~20s. Raises when it can't get a cost, so the caller can audit why."""
    if not api_key:
        raise ValueError("no OpenRouter API key to read the generation's cost")
    if not request_id or not str(request_id).startswith("gen-"):
        raise ValueError(f"not an OpenRouter generation id: {request_id!r}")
    url = (f"{config.OPENROUTER_API_BASE}/v1/generation?"
           f"{urllib.parse.urlencode({'id': request_id})}")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    last = None
    for delay in (0, *_GENERATION_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = _field(json.loads(response.read().decode("utf-8")), "data")
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code in (401, 403):
                break                    # a bad key won't fix itself
            continue
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = str(exc)
            continue
        cost = _as_cost(_field(data, "usage"))
        if cost is None:
            cost = _as_cost(_field(data, "total_cost"))
        if cost is not None:
            return cost
        last = "generation record has no cost"
    raise RuntimeError(f"OpenRouter generation cost unavailable ({last})")


def _estimate_cost(model, prompt_tokens, completion_tokens):
    """LiteLLM's price-map cost for the token counts, or None if *model* isn't
    in the map (e.g. a router alias, a local model)."""
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens)
    except Exception:
        return None
    return float(prompt_cost) + float(completion_cost)


def _report_usage(usage, model, request_id=None, hidden_params=None):
    """Hand one call's tokens and cost to billing. A no-op outside a metered
    turn, so unbilled callers pay nothing for the bookkeeping.

    An OpenRouter call is priced from OpenRouter's generation record — the
    same figure its dashboard shows — looked up in the background so the reply
    isn't delayed. Any other call uses the cost in its response, else
    LiteLLM's price-map estimate."""
    if not billing.metering():
        return
    prompt = int(_field(usage, "prompt_tokens") or 0)
    completion = int(_field(usage, "completion_tokens") or 0)
    if _uses_openrouter(model):
        api_key = _openrouter_key(model)
        billing.record_deferred(
            lambda: (_openrouter_generation_cost(request_id, api_key), "provider"),
            prompt_tokens=prompt, completion_tokens=completion, model=model,
            request_id=request_id)
        return
    cost, source = _provider_cost(usage, hidden_params), "provider"
    if cost is None:
        cost, source = _estimate_cost(model, prompt, completion), "estimate"
    if cost is None:
        cost, source = 0.0, "none"
    billing.record(cost_usd=cost, cost_source=source, prompt_tokens=prompt,
                   completion_tokens=completion, model=model,
                   request_id=request_id)


def complete(system, messages, tools=None, model=None):
    """One non-streaming turn. Returns an Anthropic-shaped _Response.

    *model* overrides the configured core model for this call only; its
    credentials are resolved per-model (see config.model_credentials)."""
    model = model or config.CORE_MODEL
    oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
    response, used = _completion(
        model,
        max_tokens=config.MAX_TOKENS,
        messages=oai_messages,
        tools=_to_litellm_tools(tools) if tools else None,
    )
    _report_usage(getattr(response, "usage", None), used,
                  request_id=getattr(response, "id", None),
                  hidden_params=getattr(response, "_hidden_params", None))
    return _normalize(response)


def stream(system, messages, tools=None, on_delta=None, model=None):
    """One streaming turn. Forwards each text delta to ``on_delta(piece)`` as it
    arrives, then returns the same Anthropic-shaped _Response as complete() so
    the caller's tool loop is otherwise unchanged. Tool-call fragments are
    reassembled across chunks before the response is built.

    *model* overrides the configured core model for this call only."""
    model = model or config.CORE_MODEL
    oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
    metered = billing.metering()
    # A stream only carries usage (and OpenRouter's cost) in a final chunk
    # that LiteLLM hands back when asked to; only ask when someone will bill it.
    extra = {"stream_options": {"include_usage": True}} if metered else {}
    chunks, used = _completion(
        model,
        max_tokens=config.MAX_TOKENS,
        messages=oai_messages,
        tools=_to_litellm_tools(tools) if tools else None,
        stream=True,
        **extra,
    )

    text_parts: list[str] = []
    tool_calls: list[dict] = []        # [{id, name, args}] in arrival order
    by_index: dict = {}                # call index -> its slot in tool_calls
    finish = None
    usage = None
    request_id = None

    for chunk in chunks:
        usage = getattr(chunk, "usage", None) or usage
        request_id = request_id or getattr(chunk, "id", None)
        if not chunk.choices:          # the usage-only chunk
            continue
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
            # A call's fragments share its index — but some providers (LiteLLM's
            # ollama_chat) send every parallel call at index 0, each with its own
            # fresh id. Keyed by index alone, those calls merged into one: args
            # concatenated, the last call's name paired with the first's args.
            # So a new id at a taken index starts a new call.
            slot = by_index.get(tc.index)
            if slot is None or (tc.id and slot["id"] and tc.id != slot["id"]):
                slot = {"id": "", "name": "", "args": ""}
                tool_calls.append(slot)
                by_index[tc.index] = slot
            if tc.id:
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn:
                if fn.name:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["args"] += fn.arguments

    if metered:
        _report_usage(usage, used, request_id=request_id)

    # See _normalize: a provider may stream tool-call deltas while reporting a
    # "stop" finish_reason, so trust the accumulated tool calls themselves.
    stop_reason = "tool_use" if (finish == "tool_calls" or tool_calls) else "end_turn"
    content: list = []
    text = "".join(text_parts)
    if text:
        content.append(_TextBlock(text=text))
    for slot in tool_calls:
        content.append(_ToolUseBlock(
            id=slot["id"],
            name=slot["name"],
            input=_loads_tool_args(slot["args"]),
        ))
    return _Response(stop_reason=stop_reason, content=content)
