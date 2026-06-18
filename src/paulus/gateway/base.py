"""Base types for the Hermes-style messaging gateway."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

_CIRCUIT_THRESHOLD = 5


@dataclass
class SessionSource:
    """Identifies a unique conversation context across platforms."""
    platform: str
    chat_id: str
    user_id: str
    thread_id: str | None = None

    def key(self) -> str:
        parts = [self.platform, self.chat_id]
        if self.thread_id:
            parts.append(self.thread_id)
        return ":".join(parts)


class AdapterState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    CIRCUIT_OPEN = "circuit_open"


# Responses containing these tokens are swallowed without delivery.
SILENCE_TOKENS = frozenset({"[SILENT]", "NO_REPLY"})


class BasePlatformAdapter(ABC):
    """
    Abstract base for all messaging platform adapters.
    Subclasses declare capabilities and implement start/stop/send.
    """
    supports_voice: bool = False
    supports_images: bool = False
    supports_typing_indicator: bool = False
    supports_threads: bool = False
    supports_streaming: bool = False

    def __init__(self, runner) -> None:
        self._runner = runner
        self._state = AdapterState.STOPPED
        self._failure_count = 0

    @property
    def state(self) -> AdapterState:
        return self._state

    @abstractmethod
    async def start(self) -> None:
        """Connect to the platform and begin receiving messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect from the platform gracefully."""

    @abstractmethod
    async def send(self, source: SessionSource, text: str) -> None:
        """Deliver a text message back to the given conversation."""

    async def send_typing(self, source: SessionSource) -> None:  # noqa: B027
        """Show a typing indicator. Optional hook; no-op by default."""
        return

    def can_approve(self, user_id) -> bool:
        """Whether this user is trusted to approve high-impact actions.
        Default: no. Adapters that set ``supports_approvals`` should override."""
        return False

    def pause(self) -> None:
        if self._state == AdapterState.RUNNING:
            self._state = AdapterState.PAUSED

    def resume(self) -> None:
        if self._state in (AdapterState.PAUSED, AdapterState.CIRCUIT_OPEN):
            self._state = AdapterState.RUNNING
            self._failure_count = 0

    def _on_success(self) -> None:
        self._failure_count = 0
        if self._state == AdapterState.CIRCUIT_OPEN:
            self._state = AdapterState.RUNNING

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= _CIRCUIT_THRESHOLD:
            self._state = AdapterState.CIRCUIT_OPEN
