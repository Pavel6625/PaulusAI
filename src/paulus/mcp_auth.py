"""OAuth sign-in for HTTP MCP servers that need it, such as Notion's hosted one.

A server entry with ``"auth": "oauth"`` signs in through the MCP SDK's OAuth
client: it discovers the authorization server, registers PaulusAI as a client
(RFC 7591) and runs the authorization code flow with PKCE. That needs a person
at a browser, so it happens once, from a terminal:

    paulus-mcp-login notion

It prints a link; after approving, the browser is sent to the redirect URI
(by default http://localhost:33418/callback). Nothing has to listen there:
the page may fail to load, and the address it was sent to, which carries the
code, is pasted back. That works the same on a headless server.

The registration and tokens live in ``<data dir>/mcp_auth/<server>.json``
(0600). The agent refreshes them on its own; when it can't (the grant expired
or was revoked), the server shows as needing a new sign-in.

Notion rotates the refresh token on every refresh and revokes the whole grant
when a retired one is replayed. So a refresh always starts from the newest
tokens on disk, and two processes sharing a sign-in (the gateway and a
terminal) don't send a stale one.
"""
import asyncio
import contextlib
import json
import logging
import os
import sys
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from mcp.shared.auth_utils import calculate_token_expiry

from . import config, mcp_client

DEFAULT_REDIRECT = "http://localhost:33418/callback"


class SignInNeeded(RuntimeError):
    """The server has no usable sign-in; only paulus-mcp-login can make one."""

    def __init__(self, server):
        super().__init__(f"not signed in: run paulus-mcp-login {server}")


def token_path(server):
    return config.DATA_DIR / "mcp_auth" / f"{mcp_client._UNSAFE.sub('_', server)}.json"


class FileStorage:
    """The SDK's TokenStorage for one server: registration and tokens in one
    0600 file, plus when the access token expires, which the SDK keeps only in
    memory (without it, a restart would treat an expired token as good)."""

    def __init__(self, server):
        self.path = token_path(server)

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.path)       # a reader never sees half a file

    def _update(self, **changes):
        data = self._load()
        for key, value in changes.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        self._save(data)

    async def get_tokens(self):
        tokens = self._load().get("tokens")
        return OAuthToken.model_validate(tokens) if tokens else None

    async def set_tokens(self, tokens):
        self._update(tokens=tokens.model_dump(mode="json", exclude_none=True),
                     expires_at=calculate_token_expiry(tokens.expires_in))

    async def get_client_info(self):
        info = self._load().get("client_info")
        return OAuthClientInformationFull.model_validate(info) if info else None

    async def set_client_info(self, client_info):
        self._update(client_info=client_info.model_dump(mode="json", exclude_none=True))

    def expires_at(self):
        return self._load().get("expires_at")

    def signed_in(self):
        return bool(self._load().get("tokens"))

    def drop_tokens(self):
        self._update(tokens=None, expires_at=None)

    def forget_tokens(self, redirect_uri):
        """Drop the tokens before a new sign-in. The registration is kept, as
        registering again orphans earlier grants, unless it was made for
        another redirect URI."""
        info = self._load().get("client_info") or {}
        if redirect_uri not in (info.get("redirect_uris") or []):
            self._update(client_info=None)
        self.drop_tokens()


class _Provider(OAuthClientProvider):
    # Three of the SDK's private hooks (pinned by mcp<3); see the module docstring.

    async def _initialize(self):
        await super()._initialize()
        self.context.token_expiry_time = self.context.storage.expires_at()

    async def _refresh_token(self):
        newest = await self.context.storage.get_tokens()
        if newest is not None and newest.refresh_token:
            self.context.current_tokens = newest
        return await super()._refresh_token()

    async def _handle_refresh_response(self, response):
        if await super()._handle_refresh_response(response):
            return True
        if response.status_code in (400, 401):      # invalid_grant: the grant is gone,
            self.context.storage.drop_tokens()      # and retrying it never helps
        return False


def redirect_uri(conf):
    return conf.get("redirect_uri") or DEFAULT_REDIRECT


def provider(server, conf, show_link=None, read_answer=None):
    """An httpx Auth for *server*. Without the two handlers it never starts a
    sign-in: a server that needs one fails with SignInNeeded instead."""
    async def no_sign_in(*_):
        raise SignInNeeded(server)

    storage = FileStorage(server)
    if show_link is None and not storage.signed_in():
        raise SignInNeeded(server)
    metadata = OAuthClientMetadata(
        client_name="PaulusAI",
        redirect_uris=[redirect_uri(conf)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )
    return _Provider(server_url=conf["url"], client_metadata=metadata, storage=storage,
                     redirect_handler=show_link or no_sign_in,
                     callback_handler=read_answer or no_sign_in)


def parse_redirect(answer):
    """The authorization response in the address the browser was sent to."""
    query = parse_qs(urlparse(answer.strip()).query)
    if "error" in query:
        detail = query.get("error_description", query["error"])[0]
        raise ValueError(f"authorization refused: {detail}")
    if "code" not in query:
        raise ValueError("that address has no ?code= in it")
    return AuthorizationCodeResult(code=query["code"][0],
                                   state=query.get("state", [None])[0],
                                   iss=query.get("iss", [None])[0])


async def login(server, conf):
    """Sign in to *server* interactively. Returns how many tools it offers."""
    from mcp import Client

    FileStorage(server).forget_tokens(redirect_uri(conf))

    async def show_link(url):
        print("\nOpen this link, approve access, then copy the address your browser "
              f"lands on\n(the page itself may fail to load; that's fine):\n\n  {url}\n")

    async def read_answer():
        loop = asyncio.get_running_loop()
        return parse_redirect(await loop.run_in_executor(None, input, "Address: "))

    srv = mcp_client._Server(server, conf)
    async with contextlib.AsyncExitStack() as stack:
        auth = provider(server, conf, show_link, read_answer)
        transport = await mcp_client._transport(srv, stack, auth=auth)
        client = await stack.enter_async_context(Client(transport))
        return len((await client.list_tools()).tools)


def main(argv=None):
    """paulus-mcp-login SERVER: sign PaulusAI in to an MCP server that uses OAuth."""
    args = sys.argv[1:] if argv is None else argv
    try:
        confs = mcp_client.load_config()
    except mcp_client.ConfigError as e:
        print(e)
        return 1
    oauth = sorted(n for n, c in confs.items() if c.get("auth") == "oauth")
    if len(args) != 1 or args[0] in ("-h", "--help"):
        print("usage: paulus-mcp-login SERVER\n\nServers in "
              f"{config.MCP_CONFIG} with \"auth\": \"oauth\": {', '.join(oauth) or '(none)'}")
        return 2
    server = args[0]
    conf = confs.get(server)
    if conf is None:
        print(f"no server '{server}' in {config.MCP_CONFIG}")
        return 1
    if "error" in conf:
        print(f"{server}: {conf['error']}")
        return 1
    if conf.get("auth") != "oauth":
        print(f"{server} doesn't sign in with OAuth (its entry has no \"auth\": \"oauth\")")
        return 1
    logging.getLogger("mcp").setLevel(logging.CRITICAL)    # we report failures ourselves
    try:
        count = asyncio.run(login(server, conf))
    except (KeyboardInterrupt, Exception) as e:
        while getattr(e, "exceptions", None):              # anyio's exception groups
            e = e.exceptions[0]
        print("\ncancelled" if isinstance(e, (KeyboardInterrupt, EOFError))
              else f"sign-in failed: {mcp_client._reason(e)}")
        return 1
    print(f"Signed in to {server} ({count} tools); saved in {token_path(server)}.\n"
          "Restart PaulusAI (systemctl restart paulus-gateway) to connect it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
