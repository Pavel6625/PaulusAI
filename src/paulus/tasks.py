"""Background tasks: long jobs that run while the conversation carries on.

A turn holds the conversation until it ends, so a job that takes minutes (a
research sweep, building and testing something) used to leave the owner
staring at "typing…". start_task, or /bg in chat, hands such a job to a
background worker instead: the agent answers at once, the owner keeps
chatting, and the result arrives as its own message when the job is done.

A background task is the agent working on its own, so:
  - it acts for the owner who started it, under the same gate. Its actions are
    reviewed against the turn that started it (the owner's words, never the
    task text the agent wrote), and approval prompts still reach the chat. In
    the terminal it can't prompt (the console belongs to the conversation), so
    an action that needs an answer is denied unless auto mode or a grant
    clears it.
  - it has a step and a time budget, and the owner can /cancel it.
  - it may delegate to subagents, but can't start further tasks.
  - task records survive restarts. Running tasks become interrupted and the
    owner is notified; work is never automatically replayed. Each result is
    also logged to the owner's episodic memory.
"""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import config, security

_tasks: dict = {}                # id -> Task, hydrated from the durable store
_initialized = False
_lock = threading.Lock()
_KEEP_FINISHED = 20              # finished tasks remembered per user, for /tasks

BRIEF = """

You are now working on a BACKGROUND TASK the owner asked for. They are not \
waiting on this conversation and can't answer questions mid-task: make \
reasonable assumptions and state them. When you're done, your final message \
is sent to the owner as the task's result, so lead with the outcome and keep \
it concise."""


@dataclass
class Task:
    id: int
    user_id: object
    title: str
    instructions: str
    status: str = "running"      # running | done | failed | cancelled | timed out | stopped | interrupted
    started: float = field(default_factory=time.time)
    finished: float | None = None
    tools_used: list = field(default_factory=list)
    result: str = ""
    interruption_reported: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    timed_out: bool = False


@contextmanager
def _store():
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.MEMORY_DIR / "tasks.sqlite3")
    try:
        with db:
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL, title TEXT NOT NULL, instructions TEXT NOT NULL,
                status TEXT NOT NULL, started REAL NOT NULL, finished REAL,
                tools_used TEXT NOT NULL, result TEXT NOT NULL DEFAULT '',
                interruption_reported INTEGER NOT NULL DEFAULT 0
            )""")
            yield db
    finally:
        db.close()


def _initialize():
    """Load records once per process (call under _lock). Never replay actions."""
    global _initialized
    if _initialized:
        return
    with _store() as db:
        db.execute("""UPDATE tasks SET status = 'interrupted', finished = ?,
                      result = ? WHERE status = 'running'""",
                   (time.time(), "The process stopped before this task finished. "
                    "It has not been resumed; some actions may already have completed."))
        rows = db.execute("""SELECT id, user_id, title, instructions, status, started,
                             finished, tools_used, result, interruption_reported
                             FROM tasks""").fetchall()
    restored = {}
    for row in rows:
        task_id, uid, title, instructions, status_, started, finished, used, result, reported = row
        restored[task_id] = Task(
            id=task_id, user_id=json.loads(uid), title=title, instructions=instructions,
            status=status_, started=started, finished=finished, tools_used=json.loads(used),
            result=result, interruption_reported=bool(reported))
    _tasks.update(restored)
    _initialized = True


def report_interrupted(user_id=None, *, all_users=False):
    """Report recovered tasks after the notifier is ready; retry unsent notices
    on the next startup. Delivery is best-effort, not an exactly-once outbox."""
    with _lock:
        _initialize()
        pending = [t for t in _tasks.values()
                   if t.status == "interrupted" and not t.interruption_reported
                   and (all_users or t.user_id == user_id)]
    for task in pending:
        message = f"⚠️ Background task #{task.id} interrupted: {task.title}\n\n{task.result}"
        try:
            if _notify(task.user_id, message) is False:
                continue
            with _lock:
                with _store() as db:
                    db.execute("UPDATE tasks SET interruption_reported = 1 WHERE id = ?",
                               (task.id,))
                task.interruption_reported = True
            security.audit("task_interrupted", f"{task.user_id} #{task.id}")
        except Exception as exc:
            security.audit("task_notify_error", f"#{task.id}: {exc}")


def _print(user_id, text):
    print(f"\n{text}\n")


_notify = _print


def set_notifier(fn):
    """Where task outcomes are announced: fn(user_id, text). Return False if
    delivery fails so interruption notices can be retried after a restart.
    The gateway sends to the user's chat; the default prints (terminal CLI)."""
    global _notify
    _notify = fn


def start_from_tool(tool_input, user_id=None, model=None, review=None):
    """The start_task tool. Returns (result, is_error), like tools.execute."""
    if review is None:
        # Only the owner's own requests start tasks; an idle nudge never does.
        return "Background tasks can only be started for the owner's requests.", True
    title = " ".join(str(tool_input.get("title") or "").split())[:80]
    instructions = str(tool_input.get("instructions") or "").strip()
    if not title or not instructions:
        return "start_task needs a 'title' and 'instructions'.", True
    task, error = start(user_id, title, instructions, model=model, review=review.fork())
    if error:
        return error, True
    return (f"Started background task #{task.id} ({title}). The owner will get its "
            f"result as a separate message; tell them it's underway."), False


def start_from_owner(text, user_id=None):
    """/bg <request>: the owner's own words become the task. Returns the reply."""
    from . import agent, billing, memory
    if not config.TASKS:
        return "Background tasks are turned off (DP_TASKS=0)."
    text = text.strip()
    if not text:
        return "Usage: /bg <what to do>"
    # A /bg is a request like any message, so it is pay-gated like one. (A task
    # started by the start_task tool is part of a turn that already was.)
    allowed, block = billing.gate(user_id)
    if not allowed:
        return block
    review = agent._review_context(text, user_id)
    memory.log_episode("owner", f"/bg {text}", trust="trusted", user_id=user_id)
    task, error = start(user_id, text[:60], text, model=config.chat_model(),
                        review=review)
    if error:
        return error
    return f"Started background task #{task.id}. I'll message you when it's done."


def start(user_id, title, instructions, model=None, review=None):
    """Start a task. Returns (task, None), or (None, why it wasn't started)."""
    if not config.TASKS:
        return None, "Background tasks are turned off (DP_TASKS=0)."
    with _lock:
        _initialize()
        running = [t for t in _tasks.values()
                   if t.user_id == user_id and t.status == "running"]
        if len(running) >= config.MAX_TASKS_PER_USER:
            return None, (f"Already running {len(running)} background task(s), the "
                          f"limit; wait for one to finish or cancel one.")
        started = time.time()
        with _store() as db:
            cursor = db.execute("""INSERT INTO tasks
                (user_id, title, instructions, status, started, tools_used)
                VALUES (?, ?, ?, 'running', ?, '[]')""",
                                (json.dumps(user_id), title, instructions, started))
            task = Task(id=cursor.lastrowid, user_id=user_id, title=title,
                        instructions=instructions, started=started)
        _tasks[task.id] = task
        _forget_old(user_id)
    security.audit("task_start", f"{user_id} #{task.id} {title}")
    thread = threading.Thread(target=_run, args=(task, config.TASK_MODEL or model, review),
                              name=f"paulus-task-{task.id}", daemon=True)
    thread.start()
    return task, None


def _run(task, model, review):
    from . import agent, billing, memory, tools
    security.BACKGROUND.set(True)        # this thread's context: no console prompts
    timer = threading.Timer(config.TASK_MAX_MINUTES * 60, _time_out, args=(task,))
    timer.daemon = True
    timer.start()
    specs = tools.agent_specs(tasks=False, user_id=task.user_id)
    pay_link = ""
    try:
        system = agent._build_system(task.user_id, specs) + BRIEF
        messages = agent._history_to_messages(task.user_id)
        agent._append_user(messages, f"[Background task #{task.id}: {task.title}]\n"
                                     f"{task.instructions}")
        # A fresh thread doesn't inherit the starting turn's meter, so the task
        # meters itself: every token it spends is billed, and it stops at its
        # next model call once the balance runs out.
        with billing.metered(task.user_id, kind="task"):
            result = agent._run_tool_loop(
                system, messages, task.user_id, model=model, tools_used=task.tools_used,
                review=review, specs=specs, max_steps=config.TASK_MAX_STEPS,
                cancel=task.cancel, origin=f"background task #{task.id} ({task.title})")
        status, body = "done", result.strip() or "(no report)"
    except billing.BalanceExhausted as exc:
        status, body = "stopped", str(exc)
        pay_link = exc.block_message[len(body):]   # the embedded link, if any
    except agent.Cancelled:
        if task.timed_out:
            status = "timed out"
            body = (f"Stopped after {config.TASK_MAX_MINUTES:g} minutes, the time "
                    f"limit. Ask me to pick it up again if it's still needed.")
        else:
            status, body = "cancelled", "Cancelled."
    except Exception as exc:
        status, body = "failed", f"It failed: {exc}"
    finally:
        timer.cancel()

    with _lock:
        finished = time.time()
        with _store() as db:
            db.execute("""UPDATE tasks SET status = ?, finished = ?, tools_used = ?,
                          result = ? WHERE id = ?""",
                       (status, finished, json.dumps(task.tools_used), body, task.id))
        task.status, task.finished, task.result = status, finished, body
        _forget_old(task.user_id)
    security.audit("task_" + status.replace(" ", "_"), f"{task.user_id} #{task.id}")
    icon = {"done": "✅", "failed": "⚠️", "cancelled": "🛑", "timed out": "⏲️",
            "stopped": "💳"}[status]
    message = f"{icon} Background task #{task.id} {status}: {task.title}\n\n{body}"
    try:
        # Logged as the agent's words, so the next turn knows the outcome.
        memory.log_episode("agent", message, trust="trusted", user_id=task.user_id)
    except Exception as exc:
        security.audit("task_log_error", f"#{task.id}: {exc}")
    try:
        # The pay link rides along for the gateway to render as a button.
        _notify(task.user_id, message + pay_link)
    except Exception as exc:
        security.audit("task_notify_error", f"#{task.id}: {exc}")


def _time_out(task):
    task.timed_out = True
    task.cancel.set()


def cancel(user_id, task_id):
    """Cancel one of *user_id*'s running tasks. Returns the reply. It stops at
    its next step: a model call or command already under way runs out first."""
    try:
        task_id = int(str(task_id).lstrip("#"))
    except ValueError:
        return "Usage: /cancel <task number>"
    with _lock:
        _initialize()
        task = _tasks.get(task_id)
        if task is None or task.user_id != user_id:
            return f"No background task #{task_id}."
        if task.status != "running":
            return f"Task #{task_id} already {task.status}."
        task.cancel.set()
    security.audit("task_cancel", f"{user_id} #{task_id}")
    return f"Cancelling task #{task_id}; it stops at its next step."


def cancel_all():
    """Stop every running task (gateway shutdown)."""
    with _lock:
        running = list(_tasks.values())
    for task in running:
        task.cancel.set()


def status(user_id):
    """A listing of *user_id*'s tasks, newest first."""
    with _lock:
        _initialize()
        mine = sorted((t for t in _tasks.values() if t.user_id == user_id),
                      key=lambda t: -t.id)
    if not mine:
        return "No background tasks."
    now = time.time()
    lines = []
    for t in mine:
        minutes = ((t.finished or now) - t.started) / 60
        progress = f"{len(t.tools_used)} step(s)"
        if t.tools_used and t.status == "running":
            progress += f", last: {t.tools_used[-1]}"
        lines.append(f"#{t.id} [{t.status}] {t.title} ({minutes:.0f} min, {progress})")
    return "\n".join(lines)


def _forget_old(user_id):
    """Keep only the latest finished tasks per user (call under _lock)."""
    finished = sorted((t for t in _tasks.values()
                       if t.user_id == user_id and t.status != "running"),
                      key=lambda t: t.id)
    # Keep interruption notices until delivery, even when newer tasks finish.
    old = [t for t in finished[:-_KEEP_FINISHED]
           if t.status != "interrupted" or t.interruption_reported]
    if old:
        with _store() as db:
            db.executemany("DELETE FROM tasks WHERE id = ?", [(t.id,) for t in old])
        for t in old:
            _tasks.pop(t.id, None)
