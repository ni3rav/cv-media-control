"""Player and system-audio interfaces. Backends register themselves elsewhere."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class Capability(Enum):
    PLAY_PAUSE = "play_pause"
    NEXT = "next"
    PREVIOUS = "previous"
    SEEK = "seek"
    VOLUME = "volume"
    MUTE = "mute"
    PITCH = "pitch"
    OPEN = "open"


def pitch_scale(semitones: float) -> float:
    """Frequency multiplier for a semitone offset. Tempo stays 1."""
    return 2.0 ** (semitones / 12.0)


class Player(Protocol):
    name: str
    capabilities: frozenset[Capability]

    def is_available(self) -> bool:
        """Return whether this backend can accept commands right now."""

    def play_pause(self) -> None:
        """Toggle playback."""

    def next(self) -> None:
        """Skip to the next track."""

    def previous(self) -> None:
        """Go to the previous track."""

    def seek(self, seconds: float) -> None:
        """Seek relative to the current position. Positive is forward."""

    def set_pitch(self, semitones: float) -> None:
        """Set pitch in semitones without changing tempo, when supported."""

    def open(self) -> None:
        """Create or raise the player window, or launch `open_command`."""

    def close(self) -> None:
        """Release the player. Must be safe to call more than once."""

    def get_position(self) -> float | None:
        """Current position in seconds, or None when the backend cannot say."""


class AudioControl(Protocol):
    def volume_delta(self, percent: float) -> None:
        """Change volume by `percent` points. Positive is louder."""

    def toggle_mute(self) -> None:
        """Toggle mute."""

    def get_state(self) -> tuple[float, bool]:
        """Return `(volume_percent, muted)`. Volume is 0 to 100."""
