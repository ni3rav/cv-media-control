"""Data the gesture engine returns. The engine class itself lives in engine.py.

`GestureEngine` in `gesture_media.engine` must provide:

- `__init__(self, config: AppConfig)`
- `step(self, hands: tuple[Observation, ...], t_ms: int) -> StepResult`
- `set_detection_enabled(self, enabled: bool) -> bool` returning the new value
- `set_paused(self, paused: bool) -> bool` returning True when the state changed
- properties `paused: bool`, `detection_enabled: bool`, `pitch_semitones: float`

`step` is a pure function of its inputs plus engine memory. Time comes only
from `t_ms` (milliseconds, non-decreasing). No clock reads and no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from gesture_media.action_types import Action, Kind


@dataclass(frozen=True, slots=True)
class ContinuousEngagement:
    """Lifecycle of one volume, seek, or pitch gesture.

    Present on ticks while the gesture is held and on the single step where it
    ends (`ended=True`). `changed` is True when a non-zero delta was emitted
    at any point during this engagement. `accumulated` is the sum of seek
    seconds or volume percent requested. For pitch, `absolute` is the current
    semitone setting and `accumulated` is unused (0).
    """

    kind: Kind
    gesture: str
    ended: bool
    changed: bool
    accumulated: float
    absolute: float | None = None


@dataclass(frozen=True, slots=True)
class StepResult:
    actions: tuple[Action, ...]
    paused: bool
    pause_toggled: bool
    detection_enabled: bool
    active_gesture: str | None
    active_score: float | None
    hold_progress: float
    engagement: ContinuousEngagement | None
    pitch_semitones: float
