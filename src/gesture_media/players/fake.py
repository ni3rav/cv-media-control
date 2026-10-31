"""In-memory player and audio control for tests and replay."""

from __future__ import annotations

from gesture_media.players.base import Capability


class FakePlayer:
    """Records every call. Seek accumulates position. Pitch is stored as set."""

    name = "fake"
    capabilities = frozenset(Capability)

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.pitch = 0.0
        self._position = 0.0

    def is_available(self) -> bool:
        self.calls.append(("is_available",))
        return True

    def play_pause(self) -> None:
        self.calls.append(("play_pause",))

    def next(self) -> None:
        self.calls.append(("next",))

    def previous(self) -> None:
        self.calls.append(("previous",))

    def seek(self, seconds: float) -> None:
        self.calls.append(("seek", seconds))
        self._position += seconds

    def set_pitch(self, semitones: float) -> None:
        self.calls.append(("set_pitch", semitones))
        self.pitch = semitones

    def open(self) -> None:
        self.calls.append(("open",))

    def close(self) -> None:
        self.calls.append(("close",))

    def get_position(self) -> float:
        self.calls.append(("get_position",))
        return self._position


class FakeAudio:
    """System-volume fake. Volume starts at 50 and is clamped to 0..100."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._volume = 50.0
        self._muted = False

    def volume_delta(self, percent: float) -> None:
        self.calls.append(("volume_delta", percent))
        self._volume = min(100.0, max(0.0, self._volume + percent))

    def toggle_mute(self) -> None:
        self.calls.append(("toggle_mute",))
        self._muted = not self._muted

    def get_state(self) -> tuple[float, bool]:
        self.calls.append(("get_state",))
        return (self._volume, self._muted)
