"""Action values passed from the gesture engine to the dispatcher.

Do not give this module I/O. The engine emits these values. The dispatcher
performs them on a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    PLAY_PAUSE = "play_pause"
    NEXT = "next"
    PREVIOUS = "previous"
    SEEK = "seek"
    VOLUME = "volume"
    MUTE_TOGGLE = "mute_toggle"
    PITCH = "pitch"
    PITCH_RESET = "pitch_reset"
    OPEN_PLAYER = "open_player"


CONTINUOUS_KINDS = frozenset({Kind.SEEK, Kind.VOLUME, Kind.PITCH})


@dataclass(frozen=True, slots=True)
class Action:
    """One player or audio operation.

    `value` depends on `kind`:
    - SEEK: delta seconds (positive is forward)
    - VOLUME: delta percent (positive is louder)
    - PITCH: absolute semitones to set, already quantized
    - everything else: None
    """

    kind: Kind
    gesture: str
    value: float | None = None

    def __post_init__(self) -> None:
        if self.kind in CONTINUOUS_KINDS and self.value is None:
            raise ValueError(f"{self.kind} requires a value")
        if self.kind not in CONTINUOUS_KINDS and self.value is not None:
            raise ValueError(f"{self.kind} does not take a value")
