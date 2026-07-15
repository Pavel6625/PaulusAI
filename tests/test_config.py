from paulus import config


def _override(monkeypatch, core="ollama_chat/gemma4:31b-cloud",
              base="https://ollama.example/v1", key="ollama-secret"):
    """Reproduce the production shape: a core model behind a custom endpoint,
    configured via the single global DP_API_BASE/DP_API_KEY pair."""
    monkeypatch.setattr(config, "CORE_MODEL", core)
    monkeypatch.setattr(config, "API_BASE", base)
    monkeypatch.setattr(config, "API_KEY", key)


def test_global_override_applies_to_core_model(monkeypatch):
    _override(monkeypatch)
    assert config.model_credentials("ollama_chat/gemma4:31b-cloud") == {
        "api_base": "https://ollama.example/v1",
        "api_key": "ollama-secret",
    }


def test_global_override_is_not_leaked_to_other_models(monkeypatch):
    # The regression this scoping exists to prevent: handing the core model's
    # endpoint and key to a different provider would break every call to it.
    _override(monkeypatch)
    creds = config.model_credentials("openrouter/anthropic/claude-sonnet-4-6")
    assert creds == {"api_base": None, "api_key": None}


def test_unset_override_yields_none_so_litellm_reads_the_env(monkeypatch):
    # With no explicit override, even CORE_MODEL must pass None rather than "",
    # so LiteLLM falls back to the provider's own env var (ANTHROPIC_API_KEY etc).
    monkeypatch.setattr(config, "CORE_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.setattr(config, "API_BASE", None)
    monkeypatch.setattr(config, "API_KEY", None)
    assert config.model_credentials("anthropic/claude-sonnet-4-6") == {
        "api_base": None, "api_key": None,
    }
