"""Web access: search the web and fetch/scrape a page into readable text.

Both entry points reach outside the machine, so they are read-only and their
results are treated as UNTRUSTED data by the caller (wrapped in tools.py). Two
concerns are handled here:

  1. **SSRF** — a fetched URL is validated (http/https only) and its host is
     resolved and rejected if it points at a loopback / private / link-local /
     reserved address, so the agent can't be steered into the local network or
     a cloud metadata endpoint (169.254.169.254).
  2. **Graceful degradation** — search works with no API key via DuckDuckGo's
     HTML endpoint; set TAVILY_API_KEY or BRAVE_API_KEY for a higher-quality
     keyed provider. HTML->text extraction uses BeautifulSoup when the `web`
     extra is installed and falls back to a stdlib stripper otherwise.
"""
import html
import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from . import config


class WebError(Exception):
    """Raised for user-facing web failures (bad URL, blocked host, HTTP error)."""


# --- HTTP -------------------------------------------------------------------

def _get(url, data=None, headers=None, timeout=None):
    """Perform an HTTP(S) request and return the decoded body as text."""
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": config.WEB_USER_AGENT, **(headers or {})},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout or config.WEB_TIMEOUT) as resp:
        raw = resp.read(config.WEB_MAX_BYTES + 1)
        if len(raw) > config.WEB_MAX_BYTES:
            raw = raw[: config.WEB_MAX_BYTES]
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


# --- SSRF guard -------------------------------------------------------------

def _validate_url(url):
    """Return a safe, absolute http(s) URL or raise WebError. Rejects hosts that
    resolve to loopback / private / link-local / reserved addresses (SSRF)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise WebError(f"Only http/https URLs are allowed, not {parsed.scheme!r}.")
    host = parsed.hostname
    if not host:
        raise WebError("URL has no host.")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise WebError(f"Could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise WebError(f"Refusing to fetch a non-public address ({addr}) for host {host!r}.")
    return url


# --- HTML -> text -----------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal stdlib fallback: collect visible text, dropping script/style and
    other non-content elements. Not as clean as BeautifulSoup but dependency-free."""

    _SKIP = {"script", "style", "head", "noscript", "template", "svg"}
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "article", "section", "header", "footer"}

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def text(self):
        joined = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", joined)).strip()


def _html_to_text(markup):
    """Extract readable text from HTML, preferring BeautifulSoup if installed."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        parser = _TextExtractor()
        parser.feed(markup)
        return parser.text()
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "head", "noscript", "template", "svg"]):
        tag.decompose()
    text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def fetch(url, max_chars=None):
    """Fetch a web page and return its readable text content. Raises WebError on
    a bad/blocked URL or a transport failure."""
    safe = _validate_url(url)
    markup = _get(safe)
    text = _html_to_text(markup)
    limit = max_chars or config.WEB_MAX_CHARS
    if len(text) > limit:
        text = text[:limit] + f"\n\n[... truncated at {limit} characters ...]"
    return text or "(no readable text extracted from this page)"


# --- Search -----------------------------------------------------------------

def _provider():
    """Resolve the active search provider. Explicit DP_SEARCH_PROVIDER wins;
    otherwise auto-detect from whichever API key is present, else DuckDuckGo."""
    import os

    explicit = (config.SEARCH_PROVIDER or "").strip().lower()
    if explicit and explicit != "auto":
        return explicit
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if os.environ.get("BRAVE_API_KEY"):
        return "brave"
    return "duckduckgo"


def _search_tavily(query, max_results):
    import os

    body = json.dumps({
        "api_key": os.environ["TAVILY_API_KEY"],
        "query": query,
        "max_results": max_results,
    }).encode()
    raw = _get("https://api.tavily.com/search", data=body,
               headers={"Content-Type": "application/json"})
    data = json.loads(raw)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""),
         "snippet": (r.get("content", "") or "").strip()}
        for r in data.get("results", [])[:max_results]
    ]


def _search_brave(query, max_results):
    import os

    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": max_results})
    raw = _get(url, headers={"Accept": "application/json",
                             "X-Subscription-Token": os.environ["BRAVE_API_KEY"]})
    data = json.loads(raw)
    results = (data.get("web", {}) or {}).get("results", [])
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""),
         "snippet": (r.get("description", "") or "").strip()}
        for r in results[:max_results]
    ]


# DuckDuckGo HTML result anchors and snippets (no API key needed).
_DDG_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL)
_DDG_SNIPPET = re.compile(
    r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _strip_tags(markup):
    return html.unescape(_TAG.sub("", markup)).strip()


def _ddg_unwrap(href):
    """DuckDuckGo wraps result links as /l/?uddg=<encoded-target>; unwrap them."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    return href


def _search_duckduckgo(query, max_results):
    body = urllib.parse.urlencode({"q": query}).encode()
    markup = _get("https://html.duckduckgo.com/html/", data=body,
                  headers={"Content-Type": "application/x-www-form-urlencoded"})
    snippets = _DDG_SNIPPET.findall(markup)
    results = []
    for i, m in enumerate(_DDG_RESULT.finditer(markup)):
        if len(results) >= max_results:
            break
        results.append({
            "title": _strip_tags(m.group("title")),
            "url": _ddg_unwrap(m.group("href")),
            "snippet": _strip_tags(snippets[i]) if i < len(snippets) else "",
        })
    return results


def search(query, max_results=None):
    """Search the web and return a list of {title, url, snippet}. Uses the keyed
    provider when configured, else DuckDuckGo's keyless HTML endpoint."""
    n = max_results or config.WEB_MAX_RESULTS
    provider = _provider()
    if provider == "tavily":
        return _search_tavily(query, n)
    if provider == "brave":
        return _search_brave(query, n)
    return _search_duckduckgo(query, n)
