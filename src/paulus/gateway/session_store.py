"""Per-conversation session metadata store."""
import time
from dataclasses import dataclass, field


@dataclass
class Session:
    key: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()

    def is_idle(self, timeout: int) -> bool:
        return (time.time() - self.last_active) > timeout


class SessionStore:
    """
    Tracks per-chat session metadata (timestamps for idle/reset logic).
    The actual message history lives in PaulusAI's episodic memory.
    """
    def __init__(self, idle_timeout: int = 3600) -> None:
        self._sessions: dict[str, Session] = {}
        self._idle_timeout = idle_timeout

    def get_or_create(self, key: str) -> Session:
        session = self._sessions.get(key)
        if session is None or session.is_idle(self._idle_timeout):
            session = Session(key=key)
            self._sessions[key] = session
        return session

    def touch(self, key: str) -> None:
        session = self._sessions.get(key)
        if session:
            session.touch()

    def reset(self, key: str) -> None:
        self._sessions.pop(key, None)
