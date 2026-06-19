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
