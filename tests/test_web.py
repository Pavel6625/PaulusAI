import pytest

from paulus import tools, web

# --- SSRF guard -------------------------------------------------------------

def test_validate_url_rejects_non_http():
    with pytest.raises(web.WebError):
        web._validate_url("ftp://example.com/x")
    with pytest.raises(web.WebError):
        web._validate_url("file:///etc/passwd")


def test_validate_url_rejects_loopback(monkeypatch):
    # Pretend the host resolves to loopback; the guard must refuse it.
    monkeypatch.setattr(web.socket, "getaddrinfo",
                        lambda *a, **k: [(0, 0, 0, "", ("127.0.0.1", 80))])
    with pytest.raises(web.WebError):
        web._validate_url("http://sneaky.example/")


def test_validate_url_rejects_metadata_endpoint(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo",
                        lambda *a, **k: [(0, 0, 0, "", ("169.254.169.254", 80))])
    with pytest.raises(web.WebError):
        web._validate_url("http://metadata/")


def test_validate_url_allows_public(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo",
                        lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 80))])
    assert web._validate_url("http://example.com/page") == "http://example.com/page"


# --- HTML -> text -----------------------------------------------------------

def test_html_to_text_drops_scripts_and_styles():
    markup = (
        "<html><head><style>.a{color:red}</style></head>"
        "<body><script>evil()</script><p>Hello</p><p>World</p></body></html>"
    )
    text = web._html_to_text(markup)
    assert "Hello" in text and "World" in text
    assert "evil" not in text and "color:red" not in text


# --- fetch ------------------------------------------------------------------

def test_fetch_truncates(monkeypatch):
    monkeypatch.setattr(web, "_validate_url", lambda u: u)
    monkeypatch.setattr(web, "_get", lambda url, **k: "<p>" + ("x" * 500) + "</p>")
    text = web.fetch("http://example.com", max_chars=100)
    assert "truncated" in text
    assert len(text) < 300


# --- search (DuckDuckGo parsing) -------------------------------------------

_DDG_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">First &amp; Best</a>
  <a class="result__snippet">A snippet here</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.org/b">Second</a>
  <a class="result__snippet">Another snippet</a>
</div>
"""


def test_search_duckduckgo_parses_results(monkeypatch):
    monkeypatch.setattr(web.config, "SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(web, "_get", lambda url, **k: _DDG_HTML)
    results = web.search("anything", max_results=5)
    assert results[0]["title"] == "First & Best"
    assert results[0]["url"] == "https://example.com/a"   # unwrapped redirect
    assert results[0]["snippet"] == "A snippet here"
    assert results[1]["url"] == "https://example.org/b"


def test_search_respects_max_results(monkeypatch):
    monkeypatch.setattr(web.config, "SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(web, "_get", lambda url, **k: _DDG_HTML)
    assert len(web.search("q", max_results=1)) == 1


# --- tools dispatch ---------------------------------------------------------

def test_tool_web_search_wraps_untrusted(monkeypatch):
    monkeypatch.setattr(web, "search",
                        lambda q, max_results=None: [
                            {"title": "T", "url": "http://x", "snippet": "S"}])
    out, is_error = tools.execute("web_search", {"query": "hi"})
    assert not is_error
    assert "<untrusted_data" in out and "http://x" in out


def test_tool_web_search_reports_error(monkeypatch):
    def boom(q, max_results=None):
        raise web.WebError("nope")
    monkeypatch.setattr(web, "search", boom)
    out, is_error = tools.execute("web_search", {"query": "hi"})
    assert is_error and "nope" in out


def test_tool_fetch_url_wraps_untrusted(monkeypatch):
    monkeypatch.setattr(web, "fetch", lambda url, max_chars=None: "page text")
    out, is_error = tools.execute("fetch_url", {"url": "http://x"})
    assert not is_error
    assert "<untrusted_data" in out and "page text" in out


def test_tool_fetch_url_reports_error(monkeypatch):
    def boom(url, max_chars=None):
        raise web.WebError("blocked")
    monkeypatch.setattr(web, "fetch", boom)
    out, is_error = tools.execute("fetch_url", {"url": "http://127.0.0.1"})
    assert is_error and "blocked" in out
