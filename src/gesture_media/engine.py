"""Deterministic gesture state machine.

Time comes only from ``step(..., t_ms)``. The clock must be non-decreasing.
There is no I/O. The engine is not thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

from gesture_media.action_types import Action, Kind
from gesture_media.config import AppConfig, GestureSpec
from gesture_media.engine_api import ContinuousEngagement, StepResult
from gesture_media.vision.types import Observation, palm_center

_THUMB_BASE = 2
_THUMB_TIP = 4
_THUMB_MIN = 0.06
# Index, middle, ring, pinky: knuckle, middle joint, tip.
_CURL_FINGERS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
_INDEX_MCP = 5
_PINKY_MCP = 17

_CONTINUOUS_ACTIONS = frozenset({"volume", "seek", "pitch"})
_KIND = {
    "play_pause": Kind.PLAY_PAUSE,
    "mute": Kind.MUTE_TOGGLE,
    "next": Kind.NEXT,
    "previous": Kind.PREVIOUS,
    "volume": Kind.VOLUME,
    "seek": Kind.SEEK,
    "pitch": Kind.PITCH,
    "open_player": Kind.OPEN_PLAYER,
    "pitch_reset": Kind.PITCH_RESET,
}
# A horizontal sweep has to be obvious, and it has to finish before a still hold.
_SWIPE_MIN_DISP = 0.18
_SWIPE_MAX_MS = 500
_SWIPE_DOMINANCE = 1.5


def _span(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def _hitchhike(landmarks: tuple) -> bool:
    """Other fingers curled and the thumb extended clear of the palm.

    MediaPipe has no sideways-thumb class. That pose is usually ``None`` or a
    fist, so the track change has to come from the landmarks.
    """
    if len(landmarks) <= 20:
        return False
    wrist = landmarks[0]
    palm = _span(landmarks[_INDEX_MCP], landmarks[_PINKY_MCP])
    if palm < 1e-4:
        return False
    for _mcp, pip, tip in _CURL_FINGERS:
        if _span(landmarks[tip], wrist) >= _span(landmarks[pip], wrist):
            return False
    bone = _span(landmarks[_THUMB_TIP], landmarks[_THUMB_BASE])
    clearance = _span(landmarks[_THUMB_TIP], landmarks[_INDEX_MCP])
    return bone > palm * 0.35 and clearance > palm * 0.8


def _thumb_delta(landmarks: tuple) -> tuple[float, float] | None:
    if len(landmarks) <= _THUMB_TIP:
        return None
    base = landmarks[_THUMB_BASE]
    tip = landmarks[_THUMB_TIP]
    return tip.x - base.x, tip.y - base.y


def thumb_aim(hand: Observation) -> str | None:
    """Image direction of an extended thumb. Right and left change the track.

    The camera frame is mirrored, so the user's right is increasing ``x``.
    Image ``y`` grows downward. A tiny vector keeps the classifier label.
    A thumb stuck out to the side counts even when the classifier says ``None``.
    """
    delta = _thumb_delta(hand.landmarks)
    if delta is None:
        return None
    dx, dy = delta
    if max(abs(dx), abs(dy)) < _THUMB_MIN:
        return None
    sideways = abs(dx) >= abs(dy)
    if sideways and (_hitchhike(hand.landmarks) or hand.gesture in {"Thumb_Up", "Thumb_Down"}):
        return "Thumb_Right" if dx > 0 else "Thumb_Left"
    if hand.gesture not in {"Thumb_Up", "Thumb_Down"}:
        return None
    return "Thumb_Up" if dy < 0 else "Thumb_Down"


def _with_thumb_aim(hand: Observation, min_score: float) -> Observation:
    label = thumb_aim(hand)
    if label is None or label == hand.gesture:
        return hand
    score = hand.score
    if label in {"Thumb_Left", "Thumb_Right"} and hand.gesture not in {"Thumb_Up", "Thumb_Down"}:
        score = max(score, min_score)
    return Observation(label, score, hand.handedness, hand.landmarks, hand.t_ms)


@dataclass(slots=True)
class _Engagement:
    kind: Kind
    gesture: str
    anchor_x: float
    anchor_y: float
    smoothed: float
    last_tick_ms: int
    engage_ms: int
    changed: bool
    accumulated: float
    absolute: float | None
    still_blocked: bool
    still_fired: bool
    motion: str
    motion_rate: float


class GestureEngine:
    """Map qualified hand observations to actions.

    A hand counts only when its gesture is not ``None``, its score meets
    ``vision.min_score``, and its palm center is inside ``vision.zone``.
    Two-fist pause looks at every hand that passed that filter and ignores
    ``vision.hand``. Single-hand control uses the hand filter.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._gestures = config.gestures
        self._paused = False
        self._detection_enabled = True
        self._pitch = 0.0
        self._pitch_raw = 0.0
        self._last_t: int | None = None
        self._actions: list[Action] = []
        self._pause_toggled = False
        self._active: str | None = None
        self._active_score: float | None = None
        self._active_since: int | None = None
        self._fired = False
        self._grace_from: int | None = None
        self._candidate: str | None = None
        self._candidate_since: int | None = None
        self._cooldowns: dict[str, int] = {}
        self._engagement: _Engagement | None = None
        self._ended: ContinuousEngagement | None = None
        self._two_since: int | None = None
        self._two_gap: int | None = None
        self._two_locked = False
        self._two_score: float | None = None
        self._swipe_origin: tuple[int, float, float] | None = None
        self._swipe_label: str | None = None

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def detection_enabled(self) -> bool:
        return self._detection_enabled

    @property
    def pitch_semitones(self) -> float:
        return self._pitch

    def set_detection_enabled(self, enabled: bool) -> bool:
        """Set detection and return the stored value."""
        self._detection_enabled = bool(enabled)
        return self._detection_enabled

    def set_paused(self, paused: bool) -> bool:
        """Set pause. Return True only when the state changes."""
        paused = bool(paused)
        if paused == self._paused:
            return False
        self._paused = paused
        return True

    def step(self, hands: tuple[Observation, ...], t_ms: int) -> StepResult:
        """Advance the machine to ``t_ms`` with the hands visible on this sample."""
        self._check_time(t_ms)
        self._actions = []
        self._pause_toggled = False

        if self._detection_enabled:
            labeled = tuple(
                _with_thumb_aim(hand, self._config.vision.min_score) for hand in hands
            )
            qualified = [hand for hand in labeled if self._qualifies(hand)]
        else:
            qualified = []
        fists = [hand for hand in qualified if hand.gesture == "Closed_Fist"]
        two_fists = self._config.pause.gesture == "two_fists" and len(fists) >= 2
        multi = len(qualified) >= 2
        two_score = min(hand.score for hand in fists) if two_fists else None
        self._pause_toggled = self._update_two_fist(two_fists, two_score, t_ms)

        if self._paused:
            if self._active is not None:
                self._release(t_ms)
            return self._result(t_ms)

        if not self._detection_enabled:
            self._single(None, None, None, t_ms, allow_emit=False)
            return self._result(t_ms)

        if two_fists or multi:
            self._single(None, None, None, t_ms, allow_emit=False)
            return self._result(t_ms)

        primary = self._select_primary(qualified)
        if primary is None or primary.gesture not in self._gestures:
            self._single(None, None, None, t_ms, allow_emit=True)
            return self._result(t_ms)
        x, y, _z = palm_center(primary)
        self._single(primary.gesture, primary.score, (x, y), t_ms, allow_emit=True)
        return self._result(t_ms)

    def _check_time(self, t_ms: int) -> None:
        if type(t_ms) is not int:
            raise ValueError(f"t_ms must be an int, got {type(t_ms).__name__}")
        if self._last_t is not None and t_ms < self._last_t:
            raise ValueError(f"t_ms went backwards from {self._last_t} to {t_ms}")
        self._last_t = t_ms

    def _qualifies(self, hand: Observation) -> bool:
        if hand.gesture == "None" or hand.score < self._config.vision.min_score:
            return False
        x, y, _z = palm_center(hand)
        zone = self._config.vision.zone
        return zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max

    def _select_primary(self, qualified: list[Observation]) -> Observation | None:
        wanted = self._config.vision.hand
        pool = qualified
        if wanted in {"left", "right"}:
            pool = [hand for hand in qualified if hand.handedness.lower() == wanted]
        if len(pool) == 1:
            return pool[0]
        return None

    def _update_two_fist(self, present: bool, score: float | None, t: int) -> bool:
        grace = self._config.engine.grace_ms
        if present:
            self._two_gap = None
            self._two_score = score
            if self._two_locked:
                return False
            if self._two_since is None:
                self._two_since = t
            if t - self._two_since >= self._config.pause.hold_ms:
                self._paused = not self._paused
                self._two_locked = True
                return True
            return False
        if self._two_since is None and not self._two_locked:
            return False
        if self._two_gap is None:
            self._two_gap = t
        if t - self._two_gap > grace:
            self._two_since = None
            self._two_locked = False
            self._two_gap = None
            self._two_score = None
        return False

    def _single(
        self,
        label: str | None,
        score: float | None,
        palm: tuple[float, float] | None,
        t: int,
        *,
        allow_emit: bool,
    ) -> None:
        if self._active is None:
            self._consider_candidate(label, score, palm, t)
            return
        if label == self._active and palm is not None and score is not None:
            self._grace_from = None
            self._candidate = None
            self._candidate_since = None
            self._active_score = score
            self._drive_active(palm, t, matching=True, allow_emit=allow_emit)
            return
        if (
            label is not None
            and palm is not None
            and score is not None
            and label in self._gestures
        ):
            if self._cooling(label, t):
                self._candidate = None
                self._candidate_since = None
            elif self._candidate != label or self._candidate_since is None:
                self._candidate = label
                self._candidate_since = t
            elif t - self._candidate_since >= self._config.engine.stable_ms:
                self._release(t)
                self._activate(label, score, palm, t)
                return
        else:
            self._candidate = None
            self._candidate_since = None
        if self._grace_from is None:
            self._grace_from = t
        if t - self._grace_from > self._config.engine.grace_ms:
            self._release(t)
            self._consider_candidate(label, score, palm, t)
            return
        self._drive_active(palm, t, matching=False, allow_emit=allow_emit)

    def _consider_candidate(
        self,
        label: str | None,
        score: float | None,
        palm: tuple[float, float] | None,
        t: int,
    ) -> None:
        if (
            label is None
            or palm is None
            or score is None
            or label not in self._gestures
            or self._cooling(label, t)
        ):
            self._candidate = None
            self._candidate_since = None
            self._swipe_origin = None
            self._swipe_label = None
            return
        if self._candidate != label or self._candidate_since is None:
            self._candidate = label
            self._candidate_since = t
            self._swipe_origin = (t, palm[0], palm[1])
            self._swipe_label = label
            return
        if self._swipe(label, palm, t, arm=False):
            spec = self._gestures[label]
            if spec.cooldown_ms:
                self._cooldowns[label] = t + spec.cooldown_ms
            self._candidate = None
            self._candidate_since = None
            self._swipe_origin = None
            self._swipe_label = None
            return
        if t - self._candidate_since >= self._config.engine.stable_ms:
            self._activate(label, score, palm, t)

    def _cooling(self, label: str, t: int) -> bool:
        until = self._cooldowns.get(label)
        return until is not None and t < until

    def _activate(self, label: str, score: float, palm: tuple[float, float], t: int) -> None:
        spec = self._gestures[label]
        self._active = label
        self._active_score = score
        self._active_since = t
        self._fired = False
        self._grace_from = None
        self._candidate = None
        self._candidate_since = None
        if spec.action not in _CONTINUOUS_ACTIONS:
            self._engagement = None
            return
        kind = _KIND[spec.action]
        self._engagement = _Engagement(
            kind=kind,
            gesture=label,
            anchor_x=palm[0],
            anchor_y=palm[1],
            smoothed=0.0,
            last_tick_ms=t,
            engage_ms=t,
            changed=False,
            accumulated=0.0,
            absolute=self._pitch if kind is Kind.PITCH else None,
            still_blocked=False,
            still_fired=False,
            motion=spec.action,
            motion_rate=spec.max_rate or 0.0,
        )

    def _release(self, t: int) -> None:
        label = self._active
        if label is None:
            return
        spec = self._gestures.get(label)
        if self._fired and spec is not None and spec.cooldown_ms:
            self._cooldowns[label] = t + spec.cooldown_ms
        if self._engagement is not None:
            self._close_engagement()
        self._active = None
        self._active_score = None
        self._active_since = None
        self._fired = False
        self._grace_from = None
        self._swipe_origin = None
        self._swipe_label = None

    def _swipe(
        self,
        label: str,
        palm: tuple[float, float],
        t: int,
        *,
        arm: bool,
    ) -> bool:
        """Horizontal sweep. Right is next, left is previous. The picture is mirrored."""
        spec = self._gestures.get(label)
        if spec is None or spec.swipe != "horizontal" or self._fired:
            return False
        origin = self._swipe_origin
        if origin is None or self._swipe_label != label:
            self._swipe_origin = (t, palm[0], palm[1])
            self._swipe_label = label
            return False
        started, x0, y0 = origin
        if t - started > _SWIPE_MAX_MS:
            self._swipe_origin = (t, palm[0], palm[1])
            return False
        dx = palm[0] - x0
        dy = palm[1] - y0
        if abs(dx) < _SWIPE_MIN_DISP or abs(dx) < abs(dy) * _SWIPE_DOMINANCE:
            return False
        right = dx > 0
        self._actions.append(
            Action(Kind.NEXT if right else Kind.PREVIOUS, "swipe_right" if right else "swipe_left")
        )
        if arm:
            self._fired = True
        return True

    def _drive_active(
        self,
        palm: tuple[float, float] | None,
        t: int,
        *,
        matching: bool,
        allow_emit: bool,
    ) -> None:
        if self._active is None:
            return
        spec = self._gestures[self._active]
        if spec.action in _CONTINUOUS_ACTIONS:
            self._drive_continuous(spec, palm, t, matching=matching, allow_emit=allow_emit)
            return
        if (
            matching
            and palm is not None
            and allow_emit
            and self._active is not None
            and self._swipe(self._active, palm, t, arm=True)
        ):
            return
        self._drive_discrete(spec, t, allow_emit=allow_emit)

    def _drive_discrete(self, spec: GestureSpec, t: int, *, allow_emit: bool) -> None:
        if (
            self._fired
            or spec.hold_ms is None
            or self._active_since is None
            or self._active is None
        ):
            return
        if t - self._active_since < spec.hold_ms or not allow_emit:
            return
        self._fired = True
        self._actions.append(Action(_KIND[spec.action], self._active, None))

    def _drive_continuous(
        self,
        spec: GestureSpec,
        palm: tuple[float, float] | None,
        t: int,
        *,
        matching: bool,
        allow_emit: bool,
    ) -> None:
        eng = self._engagement
        if eng is None:
            return
        if spec.direction is not None and spec.action == "volume":
            self._emit_directed(spec, eng, t, allow_emit=allow_emit and matching)
            return
        if matching and palm is not None:
            action, raw, rate_limit = self._motion(spec, eng, palm)
            if eng.motion != action:
                eng.smoothed = 0.0
                eng.motion = action
                eng.kind = _KIND[action]
                eng.motion_rate = rate_limit
            alpha = self._config.engine.smoothing_alpha
            eng.smoothed = alpha * raw + (1.0 - alpha) * eng.smoothed
            if abs(eng.smoothed) > self._config.engine.deadzone:
                eng.still_blocked = True
        still_ready = (
            spec.still_action is not None
            and spec.still_ms is not None
            and not eng.still_blocked
            and not eng.still_fired
            and t - eng.engage_ms >= spec.still_ms
        )
        if still_ready and allow_emit:
            eng.still_fired = True
            self._emit_still(spec, eng)
            return
        if eng.still_fired:
            return
        tick_ms = self._config.engine.tick_ms
        if t - eng.last_tick_ms < tick_ms:
            return
        elapsed_s = (t - eng.last_tick_ms) / 1000.0
        eng.last_tick_ms = t
        # A missing or mismatched sample must not keep integrating the last offset.
        if not matching or not allow_emit or spec.max_rate is None:
            return
        rate = self._rate(eng.smoothed, eng.motion_rate)
        if rate == 0.0:
            return
        self._emit_rate(eng.motion, eng, rate, elapsed_s)

    def _emit_directed(
        self,
        spec: GestureSpec,
        eng: _Engagement,
        t: int,
        *,
        allow_emit: bool,
    ) -> None:
        tick_ms = self._config.engine.tick_ms
        if t - eng.last_tick_ms < tick_ms:
            return
        elapsed_s = (t - eng.last_tick_ms) / 1000.0
        eng.last_tick_ms = t
        if not allow_emit or spec.max_rate is None or spec.direction is None:
            return
        delta = spec.direction * spec.max_rate * elapsed_s
        self._actions.append(Action(Kind.VOLUME, eng.gesture, delta))
        eng.accumulated += delta
        eng.changed = True

    def _motion(
        self,
        spec: GestureSpec,
        eng: _Engagement,
        palm: tuple[float, float],
    ) -> tuple[str, float, float]:
        dx = palm[0] - eng.anchor_x
        dy = eng.anchor_y - palm[1]
        if spec.alt_action is None or spec.alt_max_rate is None:
            raw = dx if spec.action == "seek" else dy
            return spec.action, raw, spec.max_rate or 0.0
        if abs(dx) >= abs(dy):
            return spec.action, dx, spec.max_rate or 0.0
        return spec.alt_action, dy, spec.alt_max_rate

    def _rate(self, smoothed: float, max_rate: float) -> float:
        deadzone = self._config.engine.deadzone
        span = self._config.engine.max_disp - deadzone
        magnitude = abs(smoothed)
        if magnitude <= deadzone or span <= 0.0:
            return 0.0
        fraction = (magnitude - deadzone) / span
        fraction = min(1.0, max(0.0, fraction))
        sign = 1.0 if smoothed > 0.0 else -1.0
        return sign * fraction * max_rate

    def _emit_still(self, spec: GestureSpec, eng: _Engagement) -> None:
        gesture = self._active or eng.gesture
        if spec.still_action == "pitch_reset":
            changed = abs(self._pitch) > 1e-9 or abs(self._pitch_raw) > 1e-9
            self._pitch = 0.0
            self._pitch_raw = 0.0
            eng.absolute = 0.0
            if changed:
                eng.changed = True
            self._actions.append(Action(Kind.PITCH_RESET, gesture, None))
            return
        if spec.still_action == "open_player":
            self._actions.append(Action(Kind.OPEN_PLAYER, gesture, None))

    def _emit_rate(
        self,
        action: str,
        eng: _Engagement,
        rate: float,
        elapsed_s: float,
    ) -> None:
        delta = rate * elapsed_s
        if action == "volume":
            self._actions.append(Action(Kind.VOLUME, eng.gesture, delta))
            eng.accumulated += delta
            eng.changed = True
            return
        if action == "seek":
            self._actions.append(Action(Kind.SEEK, eng.gesture, delta))
            eng.accumulated += delta
            eng.changed = True
            return
        if action != "pitch":
            return
        limits = self._config.pitch
        self._pitch_raw = min(
            limits.max_semitones,
            max(limits.min_semitones, self._pitch_raw + delta),
        )
        quantized = self._quantize(self._pitch_raw)
        eng.absolute = quantized
        if abs(quantized - self._pitch) <= 1e-9:
            return
        self._pitch = quantized
        eng.changed = True
        self._actions.append(Action(Kind.PITCH, eng.gesture, quantized))

    def _quantize(self, value: float) -> float:
        step = self._config.pitch.step
        quantized = round(round(value / step) * step, 10)
        limits = self._config.pitch
        if quantized < limits.min_semitones:
            return float(limits.min_semitones)
        if quantized > limits.max_semitones:
            return float(limits.max_semitones)
        return float(quantized)

    def _close_engagement(self) -> None:
        eng = self._engagement
        if eng is None:
            return
        pitch = eng.kind is Kind.PITCH
        self._ended = ContinuousEngagement(
            kind=eng.kind,
            gesture=eng.gesture,
            ended=True,
            changed=eng.changed,
            accumulated=0.0 if pitch else eng.accumulated,
            absolute=eng.absolute if pitch else None,
        )
        self._engagement = None

    def _engagement_result(self) -> ContinuousEngagement | None:
        if self._ended is not None:
            ended = self._ended
            self._ended = None
            return ended
        eng = self._engagement
        if eng is None:
            return None
        pitch = eng.kind is Kind.PITCH
        return ContinuousEngagement(
            kind=eng.kind,
            gesture=eng.gesture,
            ended=False,
            changed=eng.changed,
            accumulated=0.0 if pitch else eng.accumulated,
            absolute=eng.absolute if pitch else None,
        )

    def _reported_gesture(self) -> tuple[str | None, float | None]:
        if self._active is not None:
            return self._active, self._active_score
        if self._two_since is not None and self._two_gap is None:
            return "two_fists", self._two_score
        return None, None

    def _hold_progress(self, t: int) -> float:
        if self._active is not None and self._active_since is not None:
            spec = self._gestures[self._active]
            if spec.action in _CONTINUOUS_ACTIONS:
                eng = self._engagement
                if eng is None or spec.still_ms is None:
                    return 0.0
                if eng.still_fired:
                    return 1.0
                if eng.still_blocked:
                    return 0.0
                return min(1.0, max(0.0, (t - eng.engage_ms) / spec.still_ms))
            if self._fired:
                return 1.0
            if not spec.hold_ms:
                return 0.0
            return min(1.0, max(0.0, (t - self._active_since) / spec.hold_ms))
        if self._two_since is not None and self._two_gap is None:
            if self._two_locked:
                return 1.0
            hold_ms = self._config.pause.hold_ms
            if hold_ms <= 0:
                return 0.0
            return min(1.0, max(0.0, (t - self._two_since) / hold_ms))
        return 0.0

    def _result(self, t: int) -> StepResult:
        gesture, score = self._reported_gesture()
        return StepResult(
            actions=tuple(self._actions),
            paused=self._paused,
            pause_toggled=self._pause_toggled,
            detection_enabled=self._detection_enabled,
            active_gesture=gesture,
            active_score=score,
            hold_progress=self._hold_progress(t),
            engagement=self._engagement_result(),
            pitch_semitones=self._pitch,
        )
