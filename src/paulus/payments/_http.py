"""Minimal JSON-over-HTTP GET built on the stdlib.

The rest of the codebase deliberately avoids a `requests` dependency (see
web.py), so the payment validator uses urllib too. This does NOT apply the SSRF
guard: it talks to a fixed, operator-configured TON API — not to caller-supplied
URLs.
"""
import json
import urllib.error
import urllib.request


class HTTPError(Exception):
    """A non-2xx response or a transport-level failure."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _send(req, timeout, url):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.getcode()
    except urllib.error.HTTPError as e:
        # Surface the status so callers can distinguish e.g. 404/402 from others.
        raise HTTPError(f"HTTP {e.code} for {url}", status=e.code) from e
    except urllib.error.URLError as e:
        raise HTTPError(f"request to {url} failed: {e.reason}") from e
    body = {}
    if raw:
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPError(f"non-JSON response from {url}: {e}") from e
    return body, status


def get_json(url, *, headers=None, timeout=20):
    """GET and parse a JSON body. Raises HTTPError on any failure."""
    req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    return _send(req, timeout, url)[0]


def post_json(url, payload, *, headers=None, timeout=20):
    """POST JSON and return (decoded_body, status_code). Raises HTTPError on a
    transport failure or a non-2xx status (the status is on the exception)."""
    headers = dict(headers or {})
    headers.setdefault("Content-Type", "application/json")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return _send(req, timeout, url)
