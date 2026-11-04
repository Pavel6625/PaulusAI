"""Offline regressions for Telegram flood-control recovery."""
import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from paulus.gateway.base import AdapterState, SessionSource
from paulus.gateway.runner import GatewayRunner

tg = pytest.importorskip("paulus.gateway.platforms.telegram")
SOURCE = SessionSource("telegram", "42", "7", thread_id="9")
MESSAGE = SimpleNamespace(message_id=99)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "offline-test-token")
    monkeypatch.setenv("TELEGRAM_PARSE_MODE", "plain")
    monkeypatch.setenv("TELEGRAM_TRUSTED_USERS", "7")
    monkeypatch.setattr(tg.security, "audit", Mock())
    monkeypatch.setattr(tg.asyncio, "sleep", AsyncMock())
    runner = GatewayRunner()
    result = tg.TelegramAdapter(runner)
    result._state = AdapterState.RUNNING
    runner.register("telegram", result)
    result._app = SimpleNamespace(bot=SimpleNamespace(
        send_message=AsyncMock(return_value=MESSAGE),
        edit_message_text=AsyncMock(),
        send_document=AsyncMock(return_value=MESSAGE),
        delete_message=AsyncMock(),
        send_chat_action=AsyncMock(),
    ))
    return result


@pytest.mark.parametrize("delay", [12, 1.5, timedelta(seconds=1.5)])
def test_retry_preserves_arguments_result_and_server_delay(adapter, monkeypatch, delay):
    monkeypatch.setenv("PTB_TIMEDELTA", "1")
    error = tg.RetryAfter(delay)
    operation = AsyncMock(side_effect=[error, MESSAGE])
    result = asyncio.run(tg._retry_on_flood(operation, "arg", chat_id="42"))
    assert result is MESSAGE
    assert operation.await_args_list == [call("arg", chat_id="42")] * 2
    tg.asyncio.sleep.assert_awaited_once_with(
        delay.total_seconds() if isinstance(delay, timedelta) else delay
    )


def test_repeated_flood_responses_recover(adapter):
    operation = AsyncMock(side_effect=[tg.RetryAfter(12), tg.RetryAfter(12), MESSAGE])
    assert asyncio.run(tg._retry_on_flood(operation)) is MESSAGE
    assert operation.await_count == 3
    assert tg.asyncio.sleep.await_args_list == [call(12.0), call(12.0)]


@pytest.mark.parametrize("delays, expected_sleeps", [
    ([1, 1, 1, 1], [1, 1, 1]),
    ([12, 12, 2], [12, 12]),
    ([26], []),
])
def test_retry_budget_propagates_latest_error(adapter, delays, expected_sleeps):
    errors = [tg.RetryAfter(delay) for delay in delays]
    operation = AsyncMock(side_effect=errors)
    with pytest.raises(tg.RetryAfter) as caught:
        asyncio.run(tg._retry_on_flood(operation))
    assert caught.value is errors[-1]
    assert operation.await_count == len(errors)
    assert tg.asyncio.sleep.await_args_list == [call(delay) for delay in expected_sleeps]


@pytest.mark.parametrize("error", [tg.BadRequest("invalid"), RuntimeError("broken")])
def test_other_errors_are_not_retried(adapter, error):
    operation = AsyncMock(side_effect=error)
    with pytest.raises(type(error)) as caught:
        asyncio.run(tg._retry_on_flood(operation))
    assert caught.value is error
    operation.assert_awaited_once()
    tg.asyncio.sleep.assert_not_awaited()


def test_cancellation_during_request_propagates(adapter):
    operation = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(tg._retry_on_flood(operation))
    operation.assert_awaited_once()
    tg.asyncio.sleep.assert_not_awaited()


def test_cancellation_during_backoff_stops_retry(adapter, monkeypatch):
    operation = AsyncMock(side_effect=tg.RetryAfter(12))

    async def main():
        sleeping = asyncio.Event()

        async def wait(seconds):
            sleeping.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(tg.asyncio, "sleep", wait)
        task = asyncio.create_task(tg._retry_on_flood(operation))
        await sleeping.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    operation.assert_awaited_once()


def test_chunk_retry_does_not_replay_delivered_chunks(adapter):
    bot = adapter._app.bot
    text = "a" * 3500 + "b" * 3500 + "c"
    bot.send_message.side_effect = [MESSAGE, tg.RetryAfter(12), MESSAGE, MESSAGE]
    asyncio.run(adapter.send(SOURCE, text))
    assert [c.kwargs["text"] for c in bot.send_message.await_args_list] == [
        "a" * 3500, "b" * 3500, "b" * 3500, "c"
    ]
    assert all(c.kwargs["message_thread_id"] == 9 for c in bot.send_message.await_args_list)
    assert adapter._posted == {"42": 1}


@pytest.mark.parametrize("linked", [False, True])
def test_flood_then_markdown_fallback_then_plain_retry(adapter, monkeypatch, linked):
    adapter._parse_mode = "MarkdownV2"
    monkeypatch.setattr(tg, "telegramify_markdown", SimpleNamespace(
        markdownify=lambda text: "formatted"
    ))
    bot = adapter._app.bot
    bot.send_message.side_effect = [
        tg.RetryAfter(1), tg.BadRequest("can't parse entities"), tg.RetryAfter(2), MESSAGE
    ]
    coro = (adapter.send_with_link(SOURCE, "**hello**", "Open", "https://example.com")
            if linked else adapter.send(SOURCE, "**hello**"))
    asyncio.run(coro)
    calls = bot.send_message.await_args_list
    assert [c.kwargs["text"] for c in calls] == [
        "formatted", "formatted", "**hello**", "**hello**"
    ]
    assert [c.kwargs.get("parse_mode") for c in calls] == ["MarkdownV2"] * 2 + [None] * 2
    assert calls[0] == calls[1] and calls[2] == calls[3]
    if linked:
        assert all(c.kwargs["reply_markup"] == calls[0].kwargs["reply_markup"] for c in calls)


def test_document_retry_preserves_bytes_and_filename(adapter):
    bot = adapter._app.bot
    bot.send_document.side_effect = [tg.RetryAfter(12), MESSAGE]
    asyncio.run(adapter.send_document(SOURCE, "notes.md", "Привет"))
    first, second = bot.send_document.await_args_list
    assert first == second
    document = first.kwargs["document"]
    assert document.filename == "notes.md"
    assert document.input_file_content == "Привет".encode()
    assert first.kwargs["message_thread_id"] == 9
    assert adapter._posted == {"42": 1}


def test_approval_prompt_and_expiry_retry(adapter):
    bot = adapter._app.bot
    bot.send_message.side_effect = [tg.RetryAfter(1), MESSAGE]
    bot.edit_message_text.side_effect = [tg.RetryAfter(1), None]

    async def main():
        await adapter.request_approval(SOURCE, "a1", "Approve?", grantable=True)
        assert adapter._approval_msgs == {"a1": ("42", 99)}
        await adapter.expire_approval(SOURCE, "a1")

    asyncio.run(main())
    assert adapter._approval_msgs == {}
    assert bot.send_message.await_args_list[0] == bot.send_message.await_args_list[1]
    assert bot.edit_message_text.await_args_list[0] == bot.edit_message_text.await_args_list[1]
    assert len(bot.send_message.await_args.kwargs["reply_markup"].inline_keyboard) == 2


def test_callback_verdict_edit_retries_without_resolving_twice(adapter, monkeypatch):
    resolve = Mock(return_value=True)
    monkeypatch.setattr(adapter._runner, "resolve_approval", resolve)
    query = SimpleNamespace(
        data="dpok:a1", message=SimpleNamespace(text="Approve?"),
        answer=AsyncMock(), edit_message_text=AsyncMock(side_effect=[tg.RetryAfter(1), None]),
    )
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=7))
    asyncio.run(adapter._on_callback(update, None))
    resolve.assert_called_once_with("a1", True)
    query.answer.assert_awaited_once()
    assert query.edit_message_text.await_args_list == [call("Approve?\n\n— ✅ Approved.")] * 2


def test_stream_create_and_intermediate_edit_retry(adapter):
    bot = adapter._app.bot
    bot.send_message.side_effect = [tg.RetryAfter(1), MESSAGE]
    bot.edit_message_text.side_effect = [tg.RetryAfter(1), None]

    async def main():
        message = await adapter._stream_render(SOURCE, None, "first")
        assert message is MESSAGE
        assert await adapter._stream_render(SOURCE, message, "updated") is MESSAGE

    asyncio.run(main())
    assert bot.send_message.await_count == bot.edit_message_text.await_count == 2
    assert bot.edit_message_text.await_args_list == [
        call(chat_id="42", message_id=99, text="updated")
    ] * 2


@pytest.mark.parametrize("formatted", [False, True])
def test_final_edit_not_modified_after_flood_is_success(adapter, monkeypatch, formatted):
    if formatted:
        adapter._parse_mode = "MarkdownV2"
        monkeypatch.setattr(tg, "telegramify_markdown", SimpleNamespace(markdownify=lambda s: s))
    bot = adapter._app.bot
    bot.edit_message_text.side_effect = [tg.RetryAfter(1), tg.BadRequest("Message is not modified")]
    asyncio.run(adapter._stream_finalize(SOURCE, MESSAGE, "final"))
    assert bot.edit_message_text.await_count == 2
    bot.send_message.assert_not_awaited()
    tg.security.audit.assert_not_called()


def test_final_markdown_fallback_and_extra_chunk_each_retry(adapter, monkeypatch):
    adapter._parse_mode = "MarkdownV2"
    monkeypatch.setattr(tg, "telegramify_markdown", SimpleNamespace(markdownify=lambda s: s))
    bot = adapter._app.bot
    bot.edit_message_text.side_effect = [
        tg.RetryAfter(1), tg.BadRequest("can't parse entities"), tg.RetryAfter(1), None
    ]
    bot.send_message.side_effect = [tg.RetryAfter(1), MESSAGE]
    asyncio.run(adapter._stream_finalize(SOURCE, MESSAGE, "a" * 3500 + "b"))
    assert bot.edit_message_text.await_count == 4
    assert bot.edit_message_text.await_args_list[2] == bot.edit_message_text.await_args_list[3]
    assert [c.kwargs["text"] for c in bot.send_message.await_args_list] == ["b", "b"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("recover", [False, True])
def test_gateway_only_counts_unrecovered_failures(adapter, monkeypatch, streaming, recover):
    import paulus.agent as agent

    adapter.supports_streaming = streaming
    adapter._failure_count = 4
    error = tg.RetryAfter(12 if recover else 26)
    adapter._app.bot.send_message.side_effect = [error, MESSAGE]
    monkeypatch.setattr(agent, "respond", lambda *a, **kw: "reply")
    asyncio.run(adapter._runner.handle_inbound(SOURCE, "hi"))
    if recover:
        assert adapter._failure_count == 0
        assert adapter.state == AdapterState.RUNNING
        assert not any(c.args[0] == "gateway_send_error" for c in tg.security.audit.call_args_list)
    else:
        assert adapter._failure_count == 5
        assert adapter.state == AdapterState.CIRCUIT_OPEN
        assert any(c.args[0] == "gateway_send_error" for c in tg.security.audit.call_args_list)


def test_flood_wait_coalesces_frames_and_orders_segment_and_final(adapter, monkeypatch):
    bot = adapter._app.bot
    bot.send_message.side_effect = [tg.RetryAfter(12), MESSAGE, MESSAGE]

    async def main():
        sleeping, release = asyncio.Event(), asyncio.Event()

        async def wait(seconds):
            sleeping.set()
            await release.wait()

        monkeypatch.setattr(tg.asyncio, "sleep", wait)
        sink = adapter.stream_sink(SOURCE, asyncio.get_running_loop())
        sink.feed("first")
        await sleeping.wait()
        future = sink._render_future
        for _ in range(100):
            sink.feed(".")
        assert sink._render_future is future
        first_text, sink._buf = sink._buf, ""
        close = asyncio.create_task(sink._close_segment(first_text))
        release.set()
        await asyncio.wrap_future(future)
        await close
        assert sink._message is None
        sink.feed("second")
        await asyncio.wrap_future(sink._render_future)
        await sink.finalize("second")
        assert [c.kwargs["text"] for c in bot.send_message.await_args_list] == [
            "first", "first", "second"
        ]
        assert [c.kwargs["text"] for c in bot.edit_message_text.await_args_list] == [
            first_text, "second"
        ]
        # A frame arriving after finalization cannot overwrite the final reply.
        await sink._render("stale")
        assert bot.edit_message_text.await_count == 2

    asyncio.run(main())


def test_segment_timeout_keeps_commit_serialized_without_duplicate(adapter, monkeypatch):
    monkeypatch.setattr(tg, "_SEGMENT_COMMIT_TIMEOUT", 0.01)
    bot = adapter._app.bot
    bot.edit_message_text.side_effect = [tg.RetryAfter(12), None]

    async def main():
        sleeping, release = asyncio.Event(), asyncio.Event()

        async def wait(seconds):
            sleeping.set()
            await release.wait()

        monkeypatch.setattr(tg.asyncio, "sleep", wait)
        sink = adapter.stream_sink(SOURCE, asyncio.get_running_loop())
        sink._message = MESSAGE
        sink._buf = "first"
        end = asyncio.create_task(asyncio.to_thread(sink.end_segment))
        await sleeping.wait()
        await end                       # worker timed out, commit still owns lock
        assert any(c.args[0] == "telegram_segment_error" for c in tg.security.audit.call_args_list)
        sink.feed("second")
        final = asyncio.create_task(sink.finalize("second"))
        release.set()
        await final
        await asyncio.wrap_future(sink._render_future)
        assert [c.kwargs["text"] for c in bot.send_message.await_args_list] == ["second"]
        assert [c.kwargs["text"] for c in bot.edit_message_text.await_args_list] == [
            "first", "first"
        ]

    asyncio.run(main())


def test_failed_segment_does_not_reuse_old_placeholder(adapter):
    adapter._app.bot.edit_message_text.side_effect = tg.RetryAfter(26)

    async def main():
        sink = adapter.stream_sink(SOURCE, asyncio.get_running_loop())
        sink._message = MESSAGE
        with pytest.raises(tg.RetryAfter):
            await sink._close_segment("first")
        assert sink._message is None

    asyncio.run(main())


def test_best_effort_typing_and_deletion_do_not_retry(adapter):
    bot = adapter._app.bot
    bot.send_chat_action.side_effect = tg.RetryAfter(12)
    bot.delete_message.side_effect = tg.RetryAfter(12)

    async def main():
        await adapter.send_typing(SOURCE)
        sink = adapter.stream_sink(SOURCE, asyncio.get_running_loop())
        sink._message = MESSAGE
        await sink.discard()

    asyncio.run(main())
    bot.send_chat_action.assert_awaited_once()
    bot.delete_message.assert_awaited_once()
    tg.asyncio.sleep.assert_not_awaited()
