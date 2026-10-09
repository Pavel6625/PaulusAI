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
  - tasks live in memory: a restart drops running ones. Every start and
    finish is in the audit log, and each result is logged to the owner's
    episodic memory, so the next turn knows how it went.
"""
import itertools
import threading
import time
from dataclasses import dataclass, field

from . import config, security

_ids = itertools.count(1)
_tasks: dict = {}                # id -> Task, for this process's lifetime
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
    status: str = "running"      # running | done | failed | cancelled | timed out
    started: float = field(default_factory=time.time)
    finished: float | None = None
    tools_used: list = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    timed_out: bool = False


def _print(user_id, text):
    print(f"\n{text}\n")


_notify = _print


def set_notifier(fn):
    """Where finished tasks are announced: fn(user_id, text). The gateway sends
    to the user's chat; the default prints (the terminal CLI)."""
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
    from . import agent, billing, memory, router
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
    model, _tier, _reason = router.route(text, user_id=user_id)
    review = agent._review_context(text, user_id)
    memory.log_episode("owner", f"/bg {text}", trust="trusted", user_id=user_id)
    task, error = start(user_id, text[:60], text, model=model, review=review)
    if error:
        return error
    return f"Started background task #{task.id}. I'll message you when it's done."


def start(user_id, title, instructions, model=None, review=None):
    """Start a task. Returns (task, None), or (None, why it wasn't started)."""
    if not config.TASKS:
        return None, "Background tasks are turned off (DP_TASKS=0)."
    with _lock:
        running = [t for t in _tasks.values()
                   if t.user_id == user_id and t.status == "running"]
        if len(running) >= config.MAX_TASKS_PER_USER:
            return None, (f"Already running {len(running)} background task(s), the "
                          f"limit; wait for one to finish or cancel one.")
        task = Task(id=next(_ids), user_id=user_id, title=title,
                    instructions=instructions)
        _tasks[task.id] = task
        _forget_old(user_id)
    security.audit("task_start", f"{user_id} #{task.id} {title}")
    thread = threading.Thread(target=_run, args=(task, config.TASK_MODEL or model, review),
                              name=f"paulus-task-{task.id}", daemon=True)
    thread.start()
    return task, None


def _run(task, model, review):
    from . import agent, memory, tools
    security.BACKGROUND.set(True)        # this thread's context: no console prompts
    timer = threading.Timer(config.TASK_MAX_MINUTES * 60, _time_out, args=(task,))
    timer.daemon = True
    timer.start()
    specs = tools.agent_specs(tasks=False)
    try:
        system = agent._build_system(task.user_id, specs) + BRIEF
        messages = agent._history_to_messages(task.user_id)
        agent._append_user(messages, f"[Background task #{task.id}: {task.title}]\n"
                                     f"{task.instructions}")
        result = agent._run_tool_loop(
            system, messages, task.user_id, model=model, tools_used=task.tools_used,
            review=review, specs=specs, max_steps=config.TASK_MAX_STEPS,
            cancel=task.cancel, origin=f"background task #{task.id} ({task.title})")
        status, body = "done", result.strip() or "(no report)"
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

    task.status, task.finished = status, time.time()
    security.audit("task_" + status.replace(" ", "_"), f"{task.user_id} #{task.id}")
    icon = {"done": "✅", "failed": "⚠️", "cancelled": "🛑", "timed out": "⏲️"}[status]
    message = f"{icon} Background task #{task.id} {status}: {task.title}\n\n{body}"
    try:
        # Logged as the agent's words, so the next turn knows the outcome.
        memory.log_episode("agent", message, trust="trusted", user_id=task.user_id)
    except Exception as exc:
        security.audit("task_log_error", f"#{task.id}: {exc}")
    try:
        _notify(task.user_id, message)
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
    for t in finished[:-_KEEP_FINISHED]:
        _tasks.pop(t.id, None)
