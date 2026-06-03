"""The single boundary between the agent and the language model.

Keeping every model call in one small module is deliberate: it's the seam where
you swap providers (Anthropic, a local Ollama server, etc.) without touching the
agent logic, and the one place that ever sees the API key.
"""
import os
import anthropic
import config

_client = None


def _get_client():
    global _client
    if _client is None:
        # Key comes from the environment only. It is never written to disk,
        # never placed in a prompt, and never handed to the model.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running:\n"
                "  export ANTHROPIC_API_KEY=sk-..."
            )
        _client = anthropic.Anthropic()
    return _client


def complete(system, messages, tools=None):
    """One non-streaming turn. Returns the raw SDK response object so the
    caller can inspect stop_reason and content blocks."""
    kwargs = dict(
        model=config.CORE_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=messages,
    )
    if tools:
        kwargs["tools"] = tools
    return _get_client().messages.create(**kwargs)
