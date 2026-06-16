from paulus.gateway.base import AdapterState, BasePlatformAdapter, SessionSource
from paulus.gateway.runner import GatewayRunner
from paulus.gateway.session_store import SessionStore


def test_session_source_key():
    assert SessionSource("telegram", "42", "7").key() == "telegram:42"
    assert SessionSource("telegram", "42", "7", thread_id="9").key() == "telegram:42:9"


def test_session_store_recreates_after_idle():
    store = SessionStore(idle_timeout=-1)  # always considered idle
    a = store.get_or_create("k")
    b = store.get_or_create("k")
    assert a is not b


def test_authorization_allowlist():
    runner = GatewayRunner()
    src = SessionSource("t", "1", "123")
    assert runner._is_user_authorized(src, set())        # empty allowlist = allow all
    assert runner._is_user_authorized(src, {"123"})
    assert not runner._is_user_authorized(src, {"999"})


def test_dispatch_outbound_without_adapter():
    runner = GatewayRunner()
    msg = runner.dispatch_outbound("telegram:1", "hi")
    assert "unavailable" in msg or "no active adapter" in msg


def test_circuit_breaker_opens_then_resumes():
    runner = GatewayRunner()

    class Dummy(BasePlatformAdapter):
        async def start(self): ...
        async def stop(self): ...
        async def send(self, source, text): ...

    a = Dummy(runner)
    a._state = AdapterState.RUNNING
    for _ in range(5):
        a._on_failure()
    assert a.state == AdapterState.CIRCUIT_OPEN
    a.resume()
    assert a.state == AdapterState.RUNNING
