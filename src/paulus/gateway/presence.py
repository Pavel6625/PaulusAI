"""Per-user presence tracking for proactive (idle) outreach.

Unlike :class:`SessionStore`, which is keyed by conversation
(``platform:chat_id``), presence is keyed by ``user_id`` — the same dimension
the agent's memory is scoped to. Each record also remembers the user's most
recent :class:`SessionSource` so the idle loop knows *where* to reach them.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .base import SessionSource


@dataclass
class Presence:
    user_id: str
    last_source: SessionSource           # where to reach them (newest inbound)
    last_active: float = field(default_factory=time.time)
    nudges_sent: int = 0                 # unprompted msgs since the user last spoke


class PresenceStore:
    """Tracks idleness and reachability per user.

    When constructed with a ``path``, state is persisted to JSON so the nudge
    budget and idle clocks survive a gateway restart. Without a path it is
    purely in-memory (used in tests).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._users: dict[str, Presence] = {}
        if path is not None:
            self._load()

    def touch(self, source: SessionSource) -> None:
        """Record real activity from a user; clears the nudge cap."""
        p = self._users.get(source.user_id)
        if p is None:
            self._users[source.user_id] = Presence(source.user_id, source)
        else:
            p.last_source = source       # always reach them at their latest chat
            p.last_active = time.time()
            p.nudges_sent = 0            # the user spoke, so reset the budget
        self._save()

    def mark_nudged(self, user_id: str) -> None:
        """Record that a proactive message was just delivered to this user."""
        p = self._users.get(user_id)
        if p:
            p.nudges_sent += 1
            p.last_active = time.time()  # don't immediately re-fire next tick
            self._save()

    def idle_users(self, min_idle_s: float, max_nudges: int) -> list[Presence]:
        """Users quiet longer than ``min_idle_s`` and still under the nudge cap."""
        now = time.time()
        return [
            p for p in self._users.values()
            if now - p.last_active > min_idle_s and p.nudges_sent < max_nudges
        ]

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return                       # corrupt/unreadable -> start fresh
        now = time.time()
        for uid, d in raw.items():
            try:
                src = SessionSource(**d["last_source"])
            except (KeyError, TypeError):
                continue                 # skip malformed records, keep the rest
            self._users[uid] = Presence(
                user_id=uid,
                last_source=src,
                # Reset the idle clock to now: a long downtime must not make
                # everyone instantly eligible and fire a nudge burst on startup.
                # The nudge budget IS preserved so a restart can't reset the cap.
                last_active=now,
                nudges_sent=d.get("nudges_sent", 0),
            )

    def _save(self) -> None:
        if self._path is None:
            return
        data = {
            uid: {
                "last_source": asdict(p.last_source),
                "last_active": p.last_active,
                "nudges_sent": p.nudges_sent,
            }
            for uid, p in self._users.items()
        }
        try:
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass                         # best-effort; never crash the gateway
