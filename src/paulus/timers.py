"""Durable gateway schedules. Claimed occurrences are never blindly replayed."""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from croniter import croniter

from . import config, security


@contextmanager
def _store():
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.MEMORY_DIR / "timers.sqlite3", timeout=10)
    db.row_factory = sqlite3.Row
    try:
        with db:
            db.execute("""CREATE TABLE IF NOT EXISTS timers (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                title TEXT NOT NULL, payload TEXT NOT NULL, kind TEXT NOT NULL,
                cron TEXT NOT NULL, timezone TEXT NOT NULL, due REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'scheduled', owner_request TEXT NOT NULL,
                created REAL NOT NULL, last_fired REAL, last_result TEXT NOT NULL DEFAULT ''
            )""")
            yield db
    finally:
        db.close()


def _next(cron, tz, after):
    return croniter(cron, datetime.fromtimestamp(after, ZoneInfo(tz)),
                    max_years_between_matches=8).get_next(datetime).timestamp()


def create(tool_input, user_id, owner_request):
    """Validate and persist a schedule. Only called from an owner-requested loop."""
    from .gateway.runner import get_runner
    runner = get_runner()
    if user_id is None or runner is None or runner._loop is None:
        return "Timers require a running chat gateway, not the terminal.", True
    if runner._resolve_destination("", user_id)[1] is not None:
        return "No reachable chat for this timer.", True
    try:
        title = str(tool_input.get("title") or "").strip()
        payload = str(tool_input.get("payload") or "").strip()
        kind = tool_input.get("kind", "reminder")
        cron = str(tool_input.get("cron") or "").strip()
        tz = str(tool_input.get("timezone") or "UTC")
        at = str(tool_input.get("at") or "").strip()
        if not title or len(title) > 120 or not payload or len(payload) > 16000:
            raise ValueError("Provide a title (1–120 characters) and payload (1–16000).")
        if kind not in ("reminder", "agent"):
            raise ValueError("kind must be reminder or agent.")
        if bool(cron) == bool(at):
            raise ValueError("Provide exactly one of at or cron.")
        ZoneInfo(tz)
        now = time.time()
        if cron:
            if len(cron.split()) != 5:
                raise ValueError("Use a five-field cron expression (minute hour day month weekday).")
            due = _next(cron, tz, now)
        else:
            date = datetime.fromisoformat(at.replace("Z", "+00:00"))
            if date.tzinfo is None:
                raise ValueError("at must include a UTC offset or Z.")
            due = date.timestamp()
            if due <= now:
                raise ValueError("at must be in the future.")
        with _store() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM timers WHERE user_id = ? "
                               "AND status IN ('scheduled', 'firing')",
                               (json.dumps(user_id),)).fetchone()[0]
            if count >= config.MAX_TIMERS_PER_USER:
                return "Active timer limit reached; cancel a timer first.", True
            row = db.execute("""INSERT INTO timers
                (user_id, title, payload, kind, cron, timezone, due, owner_request, created)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                             (json.dumps(user_id), title, payload, kind, cron, tz, due,
                              owner_request, now))
            timer_id = row.lastrowid
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return f"Invalid timer: {exc}", True
    security.audit("timer_create", f"{user_id} #{timer_id} {kind}")
    return f"Timer #{timer_id} scheduled: {title}; next due {_date(due)}.", False


def _date(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def listing(user_id):
    with _store() as db:
        rows = db.execute("SELECT * FROM timers WHERE user_id = ? ORDER BY id DESC LIMIT 100",
                          (json.dumps(user_id),)).fetchall()
    if not rows:
        return "No timers."
    return "\n".join(f"#{r['id']} [{r['status']}] {r['title']} ({r['kind']}, "
                     f"due {_date(r['due'])}, {r['cron'] or 'once'} {r['timezone']})"
                     f"{'; last: ' + r['last_result'] if r['last_result'] else ''}" for r in rows)


def cancel(timer_id, user_id):
    try:
        timer_id = int(timer_id)
    except (ValueError, TypeError):
        return "Provide a numeric timer_id.", True
    with _store() as db:
        changed = db.execute("UPDATE timers SET status = 'cancelled' WHERE id = ? "
                             "AND user_id = ? AND status = 'scheduled'",
                             (timer_id, json.dumps(user_id))).rowcount
    if not changed:
        return "No scheduled timer with that ID for this owner (it may already be firing).", True
    security.audit("timer_cancel", f"{user_id} #{timer_id}")
    return f"Cancelled timer #{timer_id}.", False


def recover():
    """An occurrence claimed by a previous process has an uncertain outcome."""
    with _store() as db:
        db.execute("UPDATE timers SET status = CASE WHEN cron = '' THEN 'interrupted' "
                   "ELSE 'scheduled' END, last_result = 'Interrupted; outcome unknown, not replayed.' "
                   "WHERE status = 'firing'")


def _claim(now):
    with _store() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM timers WHERE status = 'scheduled' AND due <= ? "
                         "ORDER BY due, id LIMIT 1", (now,)).fetchone()
        if row is None:
            return None
        next_due = _next(row['cron'], row['timezone'], now) if row['cron'] else row['due']
        db.execute("UPDATE timers SET status = 'firing', last_fired = ?, due = ? WHERE id = ?",
                   (now, next_due, row['id']))
        return dict(row)


def _dispatch(row, notify):
    from . import agent, billing, tasks
    uid = json.loads(row['user_id'])
    if row['kind'] == 'reminder':
        if notify(uid, f"⏰ {row['title']}\n\n{row['payload']}") is False:
            raise RuntimeError("Reminder delivery failed; not automatically retried.")
        return "Reminder delivered."
    allowed, block = billing.gate(uid)
    if not allowed:
        notify(uid, block)
        return "Skipped: insufficient balance."
    # The original owner's words, not the agent-written payload, anchor review.
    review = agent._review_context(row['owner_request'], uid)
    task, error = tasks.start(uid, row['title'], row['payload'],
                              model=config.chat_model(), review=review)
    if error:
        notify(uid, f"⏰ Timer #{row['id']} could not start: {error}")
        return f"Skipped: {error}"
    return f"Started background task #{task.id}."


def fire_due(notify):
    """Run a bounded batch off the event loop. Coalesce missed cron occurrences."""
    for _ in range(20):
        row = _claim(time.time())
        if row is None:
            break
        try:
            result = _dispatch(row, notify)
            status = 'done'
        except Exception as exc:
            result, status = str(exc), 'failed'
        with _store() as db:
            db.execute("UPDATE timers SET status = ?, last_result = ? WHERE id = ?",
                       ('scheduled' if row['cron'] else status, result, row['id']))
        security.audit("timer_fire", f"#{row['id']} {result}")
