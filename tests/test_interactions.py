"""Deferred interaction persistence and safe state transitions."""
import json
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from paulus import interactions

SOURCE = {"platform": "telegram", "chat_id": "c1", "user_id": "u1", "thread_id": None}
RESULT = {"type": "tool_result", "tool_use_id": "call-1", "content": "done"}


def new(kind="approval", **overrides):
    args = {"run_id": "task:42", "task_id": 42, "user_id": "u1", "source": SOURCE,
            "kind": kind, "tool_call_id": "call-1", "tool_name": "run_command",
            "tool_input": {"command": "test"}, "checkpoint": {"messages": ["saved"]},
            "now": 100}
    args.update(overrides)
    return interactions.create(**args)


def test_pending_persists_full_correlation_and_default_expiry():
    request = new()
    restored = interactions.get(request)
    assert restored["expires"] == 100 + 12 * 60 * 60
    assert restored["run_id"] == "task:42" and restored["task_id"] == 42
    assert restored["tool_call_id"] == "call-1"
    assert restored["tool_input"] == {"command": "test"}
    assert restored["checkpoint"] == {"messages": ["saved"]}
    assert [x["id"] for x in interactions.pending("u1", SOURCE)] == [request]
    assert interactions.pending("u2", SOURCE) == []


def test_owner_conversation_and_thread_must_match():
    request = new()
    for user, source in [("u2", SOURCE), ("u1", {**SOURCE, "chat_id": "other"}),
                         ("u1", {**SOURCE, "thread_id": "other"}),
                         ("u1", {**SOURCE, "platform": "web"})]:
        assert not interactions.answer(request, user, source, "approve", now=101)
        assert not interactions.cancel(request, user, source, now=101)
    assert interactions.get(request)["status"] == "pending"
    assert interactions.answer(request, "u1", SOURCE, "approve", now=101)


def test_only_one_response_and_one_claim_under_concurrency():
    request = new()
    barrier = threading.Barrier(8)

    def answer(_):
        barrier.wait(timeout=5)
        return interactions.answer(request, "u1", SOURCE, "approve", now=101)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(answer, range(8)))
    assert sum(results) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: interactions.claim(request, now=102), range(8)))
    assert sum(item is not None for item in claims) == 1
    assert not interactions.answer(request, "u1", SOURCE, "deny", now=103)


def test_approved_action_is_never_recovered_for_replay_after_dispatch():
    request = new()
    assert interactions.answer(request, "u1", SOURCE, "approve", now=101)
    assert interactions.claim(request, now=102)["decision"] == "approve"
    assert not interactions.finish(request, RESULT, now=103)
    assert interactions.begin_execution(request, now=104)
    assert not interactions.begin_execution(request, now=104)
    assert interactions.recover(now=105) == [request]
    assert interactions.get(request)["status"] == "unknown"
    assert interactions.claim(request, now=106) is None
    assert not interactions.finish(request, RESULT, now=106)
    with pytest.raises(sqlite3.IntegrityError):
        new()


def test_processing_before_execution_is_recovered_to_answered():
    request = new()
    interactions.answer(request, "u1", SOURCE, "approve", now=101)
    interactions.claim(request, now=102)
    assert interactions.recover(now=103) == []
    assert interactions.get(request)["status"] == "answered"
    assert interactions.claim(request, now=104)["decision"] == "approve"
    assert interactions.begin_execution(request, now=105)
    assert not interactions.finish(request, {**RESULT, "tool_use_id": "other"}, now=106)
    assert interactions.finish(request, RESULT, now=106)
    assert interactions.get(request)["tool_result"] == RESULT
    assert interactions.claim(request, now=107) is None


def test_denial_and_clarification_are_correlated_and_cannot_execute():
    denied = new()
    assert not interactions.answer(denied, "u1", SOURCE, "all", now=101)
    assert interactions.answer(denied, "u1", SOURCE, "deny", now=101)
    assert interactions.claim(denied, now=102)["decision"] == "deny"
    assert not interactions.begin_execution(denied, now=103)
    assert interactions.finish(denied, RESULT, now=103)
    question = new(kind="input")
    assert not interactions.answer(question, "u1", SOURCE, " ", now=101)
    assert interactions.answer(question, "u1", SOURCE, "Use the blue one", now=101)
    assert interactions.claim(question, now=102)["decision"] == "Use the blue one"
    assert not interactions.begin_execution(question, now=103)
    assert interactions.finish(question, RESULT, now=103)


def test_expiry_is_durable_and_never_approves():
    request = new(expires_in=30)
    assert not interactions.answer(request, "u1", SOURCE, "approve", now=130)
    assert interactions.expire(now=130) == [request]
    assert interactions.expire(now=131) == []
    assert interactions.claim(request, now=132)["decision"] is None
    assert not interactions.begin_execution(request, now=133)
    interactions.recover(now=134)
    assert interactions.get(request)["status"] == "expired"
    assert interactions.claim(request, now=135)["decision"] is None
    assert interactions.finish(request, RESULT, now=136)


def test_cancel_prevents_late_answers_and_execution():
    request = new()
    assert interactions.cancel(request, "u1", SOURCE, now=101)
    assert not interactions.cancel(request, "u1", SOURCE, now=102)
    assert not interactions.answer(request, "u1", SOURCE, "approve", now=102)
    assert interactions.claim(request, now=103) is None
    assert interactions.get(request)["status"] == "cancelled"


def test_restart_in_new_process_preserves_pending_decision_and_unknown_outcome():
    from paulus import config

    request = new()
    env = {**os.environ, "DP_DATA_DIR": str(config.DATA_DIR), "PYTHONIOENCODING": "utf-8"}
    script = """
import json
from paulus import interactions
request = REQUEST
source = SOURCE
assert interactions.answer(request, 'u1', source, 'approve', now=101)
assert interactions.claim(request, now=102)['decision'] == 'approve'
assert interactions.begin_execution(request, now=103)
print(json.dumps(interactions.get(request)['status']))
""".replace("REQUEST", repr(request)).replace("SOURCE", repr(SOURCE))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=20, check=True)
    assert json.loads(result.stdout) == "executing"
    assert interactions.recover(now=104) == [request]
    assert interactions.claim(request, now=105) is None
    assert interactions.get(request)["status"] == "unknown"


def test_invalid_request_does_not_persist():
    with pytest.raises(ValueError):
        new(source={**SOURCE, "user_id": "other"})
    with pytest.raises(ValueError):
        new(expires_in=0)
    assert interactions.pending("u1", SOURCE) == []
