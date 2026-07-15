import json
from types import SimpleNamespace

from paulus import llm


def test_to_litellm_tools_shape():
    specs = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    out = llm._to_litellm_tools(specs)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "t"
    assert out[0]["function"]["parameters"] == {"type": "object"}


def test_to_openai_messages_handles_all_shapes():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "call_1", "name": "recall", "input": {"query": "x"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "found"},
        ]},
    ]
    out = llm._to_openai_messages(messages)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["tool_calls"][0]["function"]["name"] == "recall"
    assert json.loads(out[1]["tool_calls"][0]["function"]["arguments"]) == {"query": "x"}
    assert out[2]["role"] == "tool" and out[2]["tool_call_id"] == "call_1"


def test_to_openai_messages_converts_image_blocks():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
    ]}]
    out = llm._to_openai_messages(messages)
    assert out[0]["role"] == "user"
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_to_openai_messages_text_only_stays_a_string():
    # No image -> the user turn must remain a plain string, not a parts list.
    out = llm._to_openai_messages(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    )
    assert out[0] == {"role": "user", "content": "hi"}


def _capture_completion(monkeypatch, response=None):
    """Patch litellm.completion and record the kwargs it was called with."""
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return response or SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="ok", tool_calls=None),
        )])

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    return seen


def test_complete_defaults_to_core_model_with_its_credentials(monkeypatch):
    monkeypatch.setattr(llm.config, "CORE_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.setattr(llm.config, "API_BASE", "https://proxy.example/v1")
    monkeypatch.setattr(llm.config, "API_KEY", "core-secret")
    seen = _capture_completion(monkeypatch)

    llm.complete("sys", [{"role": "user", "content": "hi"}])

    assert seen["model"] == "anthropic/claude-sonnet-4-6"
    assert seen["api_base"] == "https://proxy.example/v1"
    assert seen["api_key"] == "core-secret"


def test_complete_with_explicit_model_does_not_get_core_credentials(monkeypatch):
    monkeypatch.setattr(llm.config, "CORE_MODEL", "ollama_chat/gemma4:31b-cloud")
    monkeypatch.setattr(llm.config, "API_BASE", "https://ollama.example/v1")
    monkeypatch.setattr(llm.config, "API_KEY", "ollama-secret")
    seen = _capture_completion(monkeypatch)

    llm.complete("sys", [{"role": "user", "content": "hi"}],
                 model="openrouter/openai/gpt-4o")

    assert seen["model"] == "openrouter/openai/gpt-4o"
    assert seen["api_base"] is None
    assert seen["api_key"] is None


def test_stream_honours_explicit_model(monkeypatch):
    monkeypatch.setattr(llm.config, "CORE_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.setattr(llm.config, "API_BASE", None)
    monkeypatch.setattr(llm.config, "API_KEY", None)
    chunk = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        delta=SimpleNamespace(content="hello", tool_calls=None),
    )])
    seen = _capture_completion(monkeypatch, response=[chunk])

    pieces = []
    resp = llm.stream("sys", [{"role": "user", "content": "hi"}],
                      on_delta=pieces.append, model="openrouter/openai/gpt-4o")

    assert seen["model"] == "openrouter/openai/gpt-4o"
    assert seen["stream"] is True
    assert pieces == ["hello"]
    assert resp.content[0].text == "hello"


def test_supports_vision_checks_the_given_model(monkeypatch):
    monkeypatch.setattr(llm.config, "CORE_MODEL", "text-only/model")
    asked = []

    def fake_info(model):
        asked.append(model)
        return {"supports_vision": True}

    monkeypatch.setattr(llm.litellm, "get_model_info", fake_info)
    assert llm.supports_vision("openrouter/openai/gpt-4o") is True
    assert asked == ["openrouter/openai/gpt-4o"]


def test_supports_vision_blocks_only_explicit_false(monkeypatch):
    # Catalogued as non-vision -> block.
    monkeypatch.setattr(llm.litellm, "get_model_info",
                        lambda model: {"supports_vision": False})
    assert llm.supports_vision() is False

    # Vision-capable -> allow.
    monkeypatch.setattr(llm.litellm, "get_model_info",
                        lambda model: {"supports_vision": True})
    assert llm.supports_vision() is True

    # Unknown flag (None) -> benefit of the doubt, attempt it.
    monkeypatch.setattr(llm.litellm, "get_model_info",
                        lambda model: {"supports_vision": None})
    assert llm.supports_vision() is True


def test_supports_vision_allows_unrecognized_model(monkeypatch):
    def raise_unknown(model):
        raise Exception("model not in map")
    monkeypatch.setattr(llm.litellm, "get_model_info", raise_unknown)
    assert llm.supports_vision() is True


def test_normalize_text_and_tool_calls():
    fake = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(
            content="thinking",
            tool_calls=[SimpleNamespace(
                id="c1",
                function=SimpleNamespace(name="recall", arguments=json.dumps({"query": "q"})),
            )],
        ),
    )])
    resp = llm._normalize(fake)
    assert resp.stop_reason == "tool_use"
    tub = next(b for b in resp.content if b.type == "tool_use")
    assert tub.name == "recall" and tub.input == {"query": "q"}


def test_normalize_plain_text_is_end_turn():
    fake = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content="hello", tool_calls=None),
    )])
    resp = llm._normalize(fake)
    assert resp.stop_reason == "end_turn"
    assert resp.content[0].text == "hello"
