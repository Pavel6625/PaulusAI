"""Persistent state transitions for deferred tool interactions.

A decision is tied to a user and a conversation. An action may be dispatched
only after it has been durably marked executing; a crash at that boundary
leaves an unknown outcome, never an automatic replay.

The gateway authenticates the source and applies the adapter approval policy.
Integrators must write the pending checkpoint before presenting the request and persist the
continuation after applying the result. `recover` is only safe on single-process
startup, after all old workers have stopped.
"""
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager

from . import config

_DEFAULT_EXPIRY = 12 * 60 * 60


@contextmanager
def _store():
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.MEMORY_DIR / "tasks.sqlite3", timeout=10)
    db.row_factory = sqlite3.Row
    try:
        with db:
            db.execute("""CREATE TABLE IF NOT EXISTS deferred_interactions (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id INTEGER,
                user_id TEXT NOT NULL, source TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('approval', 'input')),
                tool_call_id TEXT NOT NULL, tool_name TEXT NOT NULL,
                tool_input TEXT NOT NULL, checkpoint TEXT NOT NULL,
                created REAL NOT NULL, expires REAL NOT NULL,
                status TEXT NOT NULL, decision TEXT, tool_result TEXT, updated REAL NOT NULL
            )""")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS deferred_active_run
                ON deferred_interactions(run_id)
                WHERE status IN ('pending', 'answered', 'expired', 'processing', 'executing', 'unknown')""")
            db.execute("""CREATE TABLE IF NOT EXISTS foreground_continuations (
                interaction_id TEXT PRIMARY KEY REFERENCES deferred_interactions(id),
                status TEXT NOT NULL, checkpoint TEXT NOT NULL, updated REAL NOT NULL
            )""")
            yield db
    finally:
        db.close()


def _object(row):
    if row is None:
        return None
    obj = dict(row)
    for key in ("source", "tool_input", "checkpoint", "tool_result"):
        if obj[key] is not None:
            obj[key] = json.loads(obj[key])
    return obj


def create(*, run_id, user_id, source, kind, tool_call_id, tool_name, tool_input,
           checkpoint, task_id=None, expires_in=_DEFAULT_EXPIRY, now=None, suspend_task=False,
           continuation_id=None):
    """Atomically persist a pending request and its complete continuation.

    The caller must not dispatch or prompt before this succeeds. The run ID is
    an application-generated durable identifier; only one request may be
    active for a run. Source includes platform, chat, user and optional thread.
    """
    if not all(isinstance(x, str) and x.strip() for x in
               (run_id, user_id, tool_call_id, tool_name)):
        raise ValueError("Run, user, tool call and tool name are required")
    if kind not in ("approval", "input"):
        raise ValueError("Invalid interaction kind")
    if not isinstance(source, dict) or str(source.get("user_id")) != user_id:
        raise ValueError("The interaction source must belong to the user")
    if not all(isinstance(source.get(k), str) and source[k] for k in ("platform", "chat_id")):
        raise ValueError("Platform and conversation are required")
    if not isinstance(tool_input, dict) or not isinstance(checkpoint, dict):
        raise ValueError("Tool arguments and checkpoint must be objects")
    expires_in = float(expires_in)
    if not 0 < expires_in <= 365 * 24 * 60 * 60:
        raise ValueError("Invalid expiry")
    stamp = time.time() if now is None else float(now)
    request_id = secrets.token_urlsafe(12)
    with _store() as db:
        if suspend_task:
            if task_id is None:
                raise ValueError("A task ID is required to suspend a task")
            updated = db.execute("UPDATE tasks SET status = 'suspended' WHERE id = ? AND status = 'running'",
                                 (task_id,)).rowcount
            if updated != 1:
                raise RuntimeError("Task is not running; cannot suspend it")
        db.execute("""INSERT INTO deferred_interactions
            (id, run_id, task_id, user_id, source, kind, tool_call_id, tool_name,
             tool_input, checkpoint, created, expires, status, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                   (request_id, run_id, task_id, user_id, json.dumps(source), kind,
                    tool_call_id, tool_name, json.dumps(tool_input), json.dumps(checkpoint),
                    stamp, stamp + expires_in, stamp))
        if continuation_id is not None:
            handed_off = db.execute("""UPDATE foreground_continuations SET status = 'completed', updated = ?
                WHERE interaction_id = ? AND status = 'running' AND EXISTS
                (SELECT 1 FROM deferred_interactions WHERE id = ? AND run_id = ? AND task_id IS NULL)""",
                (stamp, continuation_id, continuation_id, run_id)).rowcount
            if task_id is not None or handed_off != 1:
                raise RuntimeError("Foreground continuation is not claimed for this run")
    return request_id


def active_task(task_id):
    """Whether task resumption is owned by a live interaction."""
    with _store() as db:
        return bool(db.execute("SELECT 1 FROM deferred_interactions WHERE task_id = ? AND status IN "
                               "('pending', 'answered', 'expired', 'processing', 'executing', 'unknown')",
                               (task_id,)).fetchone())


def _matches_source(saved, supplied):
    return all(saved.get(k) == supplied.get(k) for k in
               ("platform", "chat_id", "user_id", "thread_id"))


def answer(request_id, user_id, source, decision, *, now=None):
    """Record one owner response, or return False for invalid/stale responses.

    Approval answers are exactly 'approve' or 'deny'; a clarification answer is
    nonempty text. A caller must separately verify the adapter's trust policy
    before allowing an approval. No action is performed here.
    """
    stamp = time.time() if now is None else float(now)
    if not isinstance(source, dict):
        return False
    with _store() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM deferred_interactions WHERE id = ?", (request_id,)).fetchone()
        obj = _object(row)
        if (obj is None or obj["user_id"] != user_id
                or not _matches_source(obj["source"], source)
                or obj["status"] != "pending" or obj["expires"] <= stamp):
            return False
        if obj["kind"] == "approval":
            if decision not in ("approve", "deny"):
                return False
        elif not isinstance(decision, str) or not decision.strip() or len(decision) > 16000:
            return False
        db.execute("UPDATE deferred_interactions SET status = 'answered', decision = ?, updated = ? "
                   "WHERE id = ? AND status = 'pending'", (decision, stamp, request_id))
        return True


def claim(request_id, *, now=None):
    """Claim a settled request once, before resuming its run."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM deferred_interactions WHERE id = ?", (request_id,)).fetchone()
        if row is None or row["status"] not in ("answered", "expired"):
            return None
        db.execute("UPDATE deferred_interactions SET status = 'processing', updated = ? WHERE id = ?",
                   (stamp, request_id))
        obj = _object(row)
        obj["status"] = "processing"
        return obj


def release_claim(request_id, *, now=None):
    """Release a claim only while no approved action has begun."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        return bool(db.execute("UPDATE deferred_interactions SET status = CASE WHEN decision IS NULL "
                               "THEN 'expired' ELSE 'answered' END, updated = ? "
                               "WHERE id = ? AND status = 'processing'", (stamp, request_id)).rowcount)


def begin_execution(request_id, *, now=None, task_checkpoint=None):
    """Write the irreversible boundary before an approved tool is invoked."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        claimed = bool(db.execute("""UPDATE deferred_interactions SET status = 'executing', updated = ?
            WHERE id = ? AND status = 'processing' AND kind = 'approval' AND decision = 'approve'""",
                                  (stamp, request_id)).rowcount)
        if claimed and task_checkpoint is not None:
            task_id = db.execute("SELECT task_id FROM deferred_interactions WHERE id = ?",
                                 (request_id,)).fetchone()[0]
            if db.execute("UPDATE task_progress SET state = ? WHERE task_id = ?",
                          (json.dumps(task_checkpoint), task_id)).rowcount != 1:
                raise RuntimeError("Task checkpoint missing")
        return claimed


def finish(request_id, tool_result, *, now=None, task_checkpoint=None, continuation=None):
    """Durably record a correlated result and settle in the same transaction.

    Keeping the result with the decision prevents a crash after an action from
    losing the result while marking the request applied.
    """
    if not isinstance(tool_result, dict) or tool_result.get("type") != "tool_result":
        raise ValueError("A tool result is required")
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        applied = bool(db.execute("""UPDATE deferred_interactions SET status = 'applied',
            tool_result = ?, updated = ? WHERE id = ? AND tool_call_id = ? AND
            (status = 'executing' OR (status = 'processing' AND
            (kind = 'input' OR decision != 'approve' OR decision IS NULL)))""",
            (json.dumps(tool_result), stamp, request_id, tool_result.get("tool_use_id"))).rowcount)
        if applied and continuation is not None:
            encoded = json.dumps(continuation)
            db.execute("UPDATE deferred_interactions SET checkpoint = ? WHERE id = ?",
                       (encoded, request_id))
            task_id = db.execute("SELECT task_id FROM deferred_interactions WHERE id = ?",
                                 (request_id,)).fetchone()[0]
            if task_id is None:
                db.execute("INSERT INTO foreground_continuations VALUES (?, 'ready', ?, ?)",
                           (request_id, encoded, stamp))
        if applied and task_checkpoint is not None:
            task_id = db.execute("SELECT task_id FROM deferred_interactions WHERE id = ?",
                                 (request_id,)).fetchone()[0]
            if task_id is None:
                raise ValueError("Interaction has no task")
            if db.execute("UPDATE task_progress SET state = ? WHERE task_id = ?",
                          (json.dumps(task_checkpoint), task_id)).rowcount != 1:
                raise RuntimeError("Task checkpoint missing")
        return applied


def claim_continuation(request_id, *, now=None):
    """Claim a ready foreground continuation exactly once per recovery epoch."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        db.execute("BEGIN IMMEDIATE")
        changed = db.execute("UPDATE foreground_continuations SET status = 'running', updated = ? "
                             "WHERE interaction_id = ? AND status = 'ready'",
                             (stamp, request_id)).rowcount
        if not changed:
            return None
        row = db.execute("""SELECT d.*, c.checkpoint AS continuation_checkpoint
            FROM deferred_interactions d JOIN foreground_continuations c ON c.interaction_id = d.id
            WHERE d.id = ?""", (request_id,)).fetchone()
        result = _object(row)
        result["checkpoint"] = json.loads(result.pop("continuation_checkpoint"))
        return result


def save_continuation(request_id, checkpoint, *, now=None):
    """Persist the running foreground loop's latest execution boundary."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        return bool(db.execute("UPDATE foreground_continuations SET checkpoint = ?, updated = ? "
                               "WHERE interaction_id = ? AND status = 'running'",
                               (json.dumps(checkpoint), stamp, request_id)).rowcount)


def release_continuation(request_id, *, now=None):
    """Release a claim when the worker has stopped and retry is safe."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        return bool(db.execute("UPDATE foreground_continuations SET status = 'ready', updated = ? "
                               "WHERE interaction_id = ? AND status = 'running'",
                               (stamp, request_id)).rowcount)


def complete_continuation(request_id, *, now=None):
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        return bool(db.execute("UPDATE foreground_continuations SET status = 'completed', updated = ? "
                               "WHERE interaction_id = ? AND status = 'running'",
                               (stamp, request_id)).rowcount)


def cancel(request_id, user_id, source, *, now=None):
    """Cancel an interaction that has not yet been claimed."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM deferred_interactions WHERE id = ?", (request_id,)).fetchone()
        obj = _object(row)
        if (obj is None or obj["user_id"] != user_id or not isinstance(source, dict)
                or not _matches_source(obj["source"], source)
                or obj["status"] not in ("pending", "answered")):
            return False
        db.execute("UPDATE deferred_interactions SET status = 'cancelled', updated = ? WHERE id = ?",
                   (stamp, request_id))
        return True


def expire(*, now=None):
    """Expire pending requests, returning their IDs for denial/notification."""
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        db.execute("BEGIN IMMEDIATE")
        ids = [r[0] for r in db.execute("SELECT id FROM deferred_interactions "
                                        "WHERE status = 'pending' AND expires <= ?", (stamp,))]
        db.execute("UPDATE deferred_interactions SET status = 'expired', updated = ? "
                   "WHERE status = 'pending' AND expires <= ?", (stamp, stamp))
        return ids


def recover(*, now=None):
    """Recover after process restart; never replay a possibly executed action.

    Must be called only when no workers from a previous process can still run.
    Returns the IDs whose execution outcomes require reconciliation.
    """
    stamp = time.time() if now is None else float(now)
    with _store() as db:
        db.execute("BEGIN IMMEDIATE")
        unknown = [r[0] for r in db.execute("SELECT id FROM deferred_interactions "
                                            "WHERE status = 'executing'")]
        db.execute("UPDATE deferred_interactions SET status = 'unknown', updated = ? "
                   "WHERE status = 'executing'", (stamp,))
        db.execute("UPDATE deferred_interactions SET status = CASE WHEN decision IS NULL "
                   "THEN 'expired' ELSE 'answered' END, updated = ? WHERE status = 'processing'",
                   (stamp,))
        db.execute("UPDATE foreground_continuations SET status = 'ready', updated = ? "
                   "WHERE status = 'running'", (stamp,))
        return unknown


def pending(user_id, source):
    """Return pending requests for exactly this owner and conversation."""
    if not isinstance(source, dict):
        return []
    with _store() as db:
        rows = db.execute("SELECT * FROM deferred_interactions WHERE user_id = ? AND status = 'pending' "
                          "ORDER BY created, id", (user_id,)).fetchall()
    return [obj for row in rows if _matches_source((obj := _object(row))["source"], source)]


def unsettled():
    """Internal startup/sweeper inventory; the caller must use each saved source."""
    with _store() as db:
        rows = db.execute("SELECT * FROM deferred_interactions WHERE status IN "
                          "('pending', 'answered', 'expired', 'unknown') ORDER BY created").fetchall()
    return [_object(row) for row in rows]


def continuation_ready(request_id):
    with _store() as db:
        return bool(db.execute("SELECT 1 FROM foreground_continuations "
                               "WHERE interaction_id = ? AND status = 'ready'",
                               (request_id,)).fetchone())


def ready_continuations():
    """List durable foreground continuations ready for a one-time claim."""
    with _store() as db:
        rows = db.execute("""SELECT d.* FROM deferred_interactions d
            JOIN foreground_continuations c ON c.interaction_id = d.id
            WHERE c.status = 'ready' ORDER BY c.updated, d.id""").fetchall()
    return [_object(row) for row in rows]


def get(request_id):
    """Read a request by internal ID. Callers must enforce authorization."""
    with _store() as db:
        return _object(db.execute("SELECT * FROM deferred_interactions WHERE id = ?",
                                  (request_id,)).fetchone())
