"""Scripted-clock tests for the gesture state machine."""

from __future__ import annotations

from dataclasses import replace

import pytest

from gesture_media.action_types import Kind
from gesture_media.config import default_config
from gesture_media.engine import GestureEngine
from gesture_media.engine_api import StepResult
from gesture_media.vision.types import Landmark, Observation

PALM = {0, 5, 9, 13, 17}


def make_hand(
    gesture: str,
    x: float,
    y: float,
    *,
    score: float = 0.95,
    handedness: str = "Right",
    t: int = 0,
) -> Observation:
    landmarks = tuple(
        Landmark(x, y, 0.0) if index in PALM else Landmark(0.0, 0.0, 0.0) for index in range(21)
    )
    return Observation(gesture, score, handedness, landmarks, t)


def feed(
    engine: GestureEngine,
    gesture: str,
    x: float,
    y: float,
    times: range,
    **kwargs: object,
) -> list[StepResult]:
    results = []
    for t in times:
        results.append(engine.step((make_hand(gesture, x, y, t=t, **kwargs),), t))
    return results


def feed_empty(engine: GestureEngine, times: range) -> list[StepResult]:
    return [engine.step((), t) for t in times]


def kinds(results: list[StepResult]) -> list[Kind]:
    return [action.kind for result in results for action in result.actions]


def with_alpha(alpha: float) -> GestureEngine:
    cfg = default_config()
    return GestureEngine(replace(cfg, engine=replace(cfg.engine, smoothing_alpha=alpha)))


@pytest.mark.parametrize(
    ("gesture", "kind", "hold_ms"),
    [
        ("Open_Palm", Kind.PLAY_PAUSE, 500),
        ("Closed_Fist", Kind.MUTE_TOGGLE, 500),
        ("Thumb_Up", Kind.NEXT, 400),
        ("Thumb_Down", Kind.PREVIOUS, 400),
    ],
)
def test_each_discrete_gesture_fires_once(gesture: str, kind: Kind, hold_ms: int) -> None:
    engine = GestureEngine(default_config())
    fire_at = 150 + hold_ms
    fired_at = None
    actions = []
    for t in range(0, fire_at + 200, 50):
        result = engine.step((make_hand(gesture, 0.5, 0.5, t=t),), t)
        if result.actions:
            fired_at = t
            actions.extend(result.actions)
        if t == fire_at:
            assert result.hold_progress == pytest.approx(1.0)
    assert fired_at == fire_at
    assert len(actions) == 1
    assert actions[0].kind is kind
    assert actions[0].gesture == gesture
    assert actions[0].value is None


@pytest.mark.parametrize(
    "gesture",
    ["Open_Palm", "Closed_Fist", "Thumb_Up", "Thumb_Down"],
)
def test_hold_too_short_does_not_fire(gesture: str) -> None:
    engine = GestureEngine(default_config())
    feed(engine, gesture, 0.5, 0.5, range(0, 151, 50))
    results = feed_empty(engine, range(200, 501, 50))
    assert kinds(results) == []
    assert results[-1].active_gesture is None


def test_short_hold_does_not_start_cooldown() -> None:
    engine = GestureEngine(default_config())
    feed(engine, "Open_Palm", 0.5, 0.5, range(0, 301, 50))
    assert kinds(feed_empty(engine, range(350, 601, 50))) == []
    results = feed(engine, "Open_Palm", 0.5, 0.5, range(650, 1401, 50))
    assert kinds(results) == [Kind.PLAY_PAUSE]


def test_flicker_is_rejected() -> None:
    engine = GestureEngine(default_config())
    engine.step((make_hand("Open_Palm", 0.5, 0.5),), 0)
    engine.step((make_hand("Thumb_Up", 0.5, 0.5),), 100)
    engine.step((make_hand("Open_Palm", 0.5, 0.5),), 200)
    early = engine.step((make_hand("Open_Palm", 0.5, 0.5),), 300)
    assert early.active_gesture is None
    assert early.actions == ()
    active = engine.step((make_hand("Open_Palm", 0.5, 0.5),), 350)
    assert active.active_gesture == "Open_Palm"
    assert active.actions == ()

    stable = GestureEngine(default_config())
    stable.step((make_hand("Open_Palm", 0.5, 0.5),), 0)
    stable.step((make_hand("Open_Palm", 0.5, 0.5),), 150)
    stable.step((make_hand("Thumb_Up", 0.5, 0.5),), 200)
    back = stable.step((make_hand("Open_Palm", 0.5, 0.5),), 250)
    assert back.active_gesture == "Open_Palm"
    fired = feed(stable, "Open_Palm", 0.5, 0.5, range(300, 701, 50))
    assert kinds(fired) == [Kind.PLAY_PAUSE]


def test_grace_period_keeps_hold_progress() -> None:
    engine = GestureEngine(default_config())
    feed(engine, "Open_Palm", 0.5, 0.5, range(0, 501, 50))
    for t in (550, 600):
        missing = engine.step((), t)
        assert missing.active_gesture == "Open_Palm"
        assert missing.actions == ()
    fired = engine.step((make_hand("Open_Palm", 0.5, 0.5),), 650)
    assert [action.kind for action in fired.actions] == [Kind.PLAY_PAUSE]
    assert fired.hold_progress == pytest.approx(1.0)


def test_lock_until_release_then_cooldown() -> None:
    engine = GestureEngine(default_config())
    held = feed(engine, "Open_Palm", 0.5, 0.5, range(0, 1501, 50))
    assert kinds(held) == [Kind.PLAY_PAUSE]
    assert held[-1].hold_progress == pytest.approx(1.0)
    released_at = None
    gaps = range(1550, 2001, 50)
    for result, t in zip(feed_empty(engine, gaps), gaps, strict=True):
        if result.active_gesture is None and released_at is None:
            released_at = t
    assert released_at == 1800
    during = feed(engine, "Open_Palm", 0.5, 0.5, range(2050, 2800, 50))
    assert kinds(during) == []
    assert all(result.active_gesture is None for result in during)
    after = feed(engine, "Open_Palm", 0.5, 0.5, range(2800, 3601, 50))
    assert kinds(after) == [Kind.PLAY_PAUSE]


def test_hand_loss_clears_locks_but_not_cooldowns_or_pitch() -> None:
    engine = with_alpha(1.0)
    pitched = _raise_pitch(engine, 0)
    assert engine.pitch_semitones > 0
    lost = feed_empty(engine, range(pitched, pitched + 400, 50))
    assert lost[-1].active_gesture is None
    assert engine.pitch_semitones > 0
    assert all(action.kind is not Kind.PITCH_RESET for result in lost for action in result.actions)

    fresh = GestureEngine(default_config())
    feed(fresh, "Closed_Fist", 0.5, 0.5, range(0, 801, 50))
    release = feed_empty(fresh, range(850, 1101, 50))
    assert release[-1].active_gesture is None
    blocked = feed(fresh, "Closed_Fist", 0.5, 0.5, range(1150, 1801, 50))
    assert kinds(blocked) == []
    assert all(result.active_gesture is None for result in blocked)
    thumb = feed(fresh, "Thumb_Up", 0.5, 0.5, range(1850, 2501, 50))
    assert Kind.NEXT in kinds(thumb)
    assert any(
        result.active_gesture == "Thumb_Up" and t < 2100 for t, result in _with_times(thumb, 1850)
    )


def _with_times(results: list[StepResult], start: int) -> list[tuple[int, StepResult]]:
    return [(start + index * 50, result) for index, result in enumerate(results)]


def test_continuous_deadzone_clamp_and_directions() -> None:
    # alpha 1 makes one tick equal the raw displacement.
    assert _delta("Pointing_Up", 0.5, 0.45) is None  # exactly the deadzone
    assert _delta("Pointing_Up", 0.5, 0.46) is None
    assert _delta("Pointing_Up", 0.5, 0.44) == pytest.approx(0.15)
    assert _delta("Pointing_Up", 0.5, 0.20) == pytest.approx(3.0)
    assert _delta("Pointing_Up", 0.5, 0.80) == pytest.approx(-3.0)
    assert _delta("Victory", 0.80, 0.5) == pytest.approx(2.0)
    assert _delta("Victory", 0.20, 0.5) == pytest.approx(-2.0)
    assert _delta("ILoveYou", 0.5, 0.20) == pytest.approx(0.5)
    assert _delta("ILoveYou", 0.5, 0.80) == pytest.approx(-0.5)


def test_default_smoothing_ramps_toward_the_clamp() -> None:
    engine = GestureEngine(default_config())
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 0)
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 150)
    values = []
    for index in range(20):
        result = engine.step((make_hand("Pointing_Up", 0.5, 0.25),), 250 + index * 100)
        values.extend(action.value for action in result.actions if action.value is not None)
    assert values[0] == pytest.approx(0.75)
    assert values[-1] == pytest.approx(3.0, abs=0.05)
    assert all(abs(value) <= 3.0 + 1e-9 for value in values)


def test_hold_still_fires_once() -> None:
    engine = GestureEngine(default_config())
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 0)
    engaged = engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 150)
    assert engaged.engagement is not None
    assert engaged.engagement.ended is False
    assert engaged.hold_progress == pytest.approx(0.0)
    midpoint = engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 900)
    assert midpoint.actions == ()
    assert midpoint.hold_progress == pytest.approx(0.5)
    fired = engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 1650)
    assert [action.kind for action in fired.actions] == [Kind.OPEN_PLAYER]
    assert fired.actions[0].value is None
    assert fired.hold_progress == pytest.approx(1.0)
    later = engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 3000)
    assert later.actions == ()


def test_hold_still_is_suppressed_after_leaving_the_deadzone() -> None:
    engine = GestureEngine(default_config())
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 0)
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 150)
    moved = engine.step((make_hand("Pointing_Up", 0.5, 0.30),), 250)
    assert any(action.kind is Kind.VOLUME for action in moved.actions)
    assert moved.engagement is not None
    assert moved.engagement.changed is True
    for t in (350, 1850, 2500, 4000):
        result = engine.step((make_hand("Pointing_Up", 0.5, 0.5),), t)
        assert all(action.kind is not Kind.OPEN_PLAYER for action in result.actions)
    ended = feed_empty(engine, range(4050, 4401, 50))
    assert any(result.engagement is not None and result.engagement.ended for result in ended)
    assert ended[-1].active_gesture is None


def test_zone_score_and_palm_center_arming() -> None:
    outside = GestureEngine(default_config())
    assert kinds(feed(outside, "Open_Palm", 0.05, 0.5, range(0, 1001, 50))) == []
    low_y = GestureEngine(default_config())
    assert kinds(feed(low_y, "Open_Palm", 0.5, 0.05, range(0, 1001, 50))) == []
    low = GestureEngine(default_config())
    assert kinds(feed(low, "Open_Palm", 0.5, 0.5, range(0, 1001, 50), score=0.59)) == []
    none_gesture = GestureEngine(default_config())
    assert kinds(feed(none_gesture, "None", 0.5, 0.5, range(0, 1001, 50))) == []

    # Non-palm landmarks are 0. Averaging every landmark would leave the zone.
    boundary = GestureEngine(default_config())
    assert kinds(feed(boundary, "Open_Palm", 0.15, 0.5, range(0, 801, 50))) == [Kind.PLAY_PAUSE]
    exact = GestureEngine(default_config())
    assert kinds(feed(exact, "Thumb_Up", 0.5, 0.5, range(0, 801, 50), score=0.6)) == [Kind.NEXT]


def test_gesture_switching() -> None:
    engine = GestureEngine(default_config())
    feed(engine, "Open_Palm", 0.5, 0.5, range(0, 151, 50))
    engine.step((make_hand("Thumb_Up", 0.5, 0.5),), 200)
    engine.step((make_hand("Thumb_Up", 0.5, 0.5),), 250)
    still_palm = engine.step((make_hand("Thumb_Up", 0.5, 0.5),), 300)
    assert still_palm.active_gesture == "Open_Palm"
    switched = engine.step((make_hand("Thumb_Up", 0.5, 0.5),), 350)
    assert switched.active_gesture == "Thumb_Up"
    assert switched.actions == ()
    fired = feed(engine, "Thumb_Up", 0.5, 0.5, range(400, 801, 50))
    assert kinds(fired) == [Kind.NEXT]


def test_two_fist_pause_and_suppression() -> None:
    engine = GestureEngine(default_config())
    toggles = []
    for t in range(0, 2501, 50):
        result = engine.step(_fists(t), t)
        if result.pause_toggled:
            toggles.append((t, result.paused))
        assert all(action.kind is not Kind.MUTE_TOGGLE for action in result.actions)
        if t == 1000:
            assert result.active_gesture == "two_fists"
            assert result.hold_progress == pytest.approx(0.5)
    assert toggles == [(2000, True)]
    assert engine.paused is True

    suppressed = feed(engine, "Open_Palm", 0.5, 0.5, range(2600, 4001, 50))
    suppressed += feed(engine, "Closed_Fist", 0.5, 0.5, range(4050, 5001, 50))
    assert kinds(suppressed) == []
    assert engine.paused is True

    feed_empty(engine, range(5050, 5401, 50))
    resume = []
    for t in range(5450, 8001, 50):
        result = engine.step(_fists(t), t)
        if result.pause_toggled:
            resume.append((t, result.paused))
    assert resume == [(7450, False)]
    assert engine.paused is False
    feed_empty(engine, range(8050, 8401, 50))
    played = feed(engine, "Open_Palm", 0.5, 0.5, range(8450, 9501, 50))
    assert kinds(played) == [Kind.PLAY_PAUSE]


def test_two_fist_hold_too_short_does_nothing() -> None:
    engine = GestureEngine(default_config())
    for t in range(0, 1501, 50):
        result = engine.step(_fists(t), t)
        assert result.pause_toggled is False
    trailing = feed_empty(engine, range(1600, 4001, 50))
    assert all(not result.pause_toggled for result in trailing)
    assert engine.paused is False


def test_single_fist_mutes_and_two_hands_suppress_single_hand_actions() -> None:
    single = GestureEngine(default_config())
    assert kinds(feed(single, "Closed_Fist", 0.5, 0.5, range(0, 801, 50))) == [Kind.MUTE_TOGGLE]

    outside = GestureEngine(default_config())
    actions = []
    for t in range(0, 801, 50):
        hands = (
            make_hand("Closed_Fist", 0.5, 0.5, t=t),
            make_hand("Closed_Fist", 0.02, 0.5, t=t),
        )
        actions.extend(action.kind for action in outside.step(hands, t).actions)
    assert actions == [Kind.MUTE_TOGGLE]
    assert outside.paused is False

    both = GestureEngine(default_config())
    mixed = []
    for t in range(0, 2001, 50):
        hands = (
            make_hand("Open_Palm", 0.4, 0.5, t=t),
            make_hand("Thumb_Up", 0.7, 0.5, t=t),
        )
        result = both.step(hands, t)
        mixed.extend(result.actions)
        assert result.pause_toggled is False
    assert mixed == []


def test_hand_filter_and_two_fist_ignores_it() -> None:
    cfg = default_config()
    cfg = replace(cfg, vision=replace(cfg.vision, hand="left"))
    filtered = GestureEngine(cfg)
    right = feed(filtered, "Open_Palm", 0.5, 0.5, range(0, 1001, 50), handedness="Right")
    assert kinds(right) == []
    left = feed(filtered, "Open_Palm", 0.5, 0.5, range(1100, 2101, 50), handedness="Left")
    assert kinds(left) == [Kind.PLAY_PAUSE]

    pause = GestureEngine(cfg)
    toggled = False
    for t in range(0, 2001, 50):
        hands = (
            make_hand("Closed_Fist", 0.35, 0.5, handedness="Left", t=t),
            make_hand("Closed_Fist", 0.65, 0.5, handedness="Right", t=t),
        )
        if pause.step(hands, t).pause_toggled:
            toggled = True
    assert toggled is True
    assert pause.paused is True


def test_pitch_quantize_clamp_and_hand_loss_keeps_it() -> None:
    engine = with_alpha(1.0)
    engine.step((make_hand("ILoveYou", 0.5, 0.5),), 0)
    engine.step((make_hand("ILoveYou", 0.5, 0.5),), 150)
    raw = 0.0
    pitch = 0.0
    expected: list[float] = []
    actual: list[float] = []
    for index in range(5):
        result = engine.step((make_hand("ILoveYou", 0.5, 0.20),), 250 + index * 100)
        raw = min(12.0, raw + 0.4)
        quantized = round(round(raw / 0.5) * 0.5, 10)
        if abs(quantized - pitch) > 1e-9:
            expected.append(quantized)
            pitch = quantized
        actual.extend(action.value for action in result.actions if action.value is not None)
        if result.engagement is not None:
            assert result.engagement.accumulated == pytest.approx(0.0)
            assert result.engagement.absolute == pytest.approx(engine.pitch_semitones)
    assert actual == pytest.approx(expected)
    assert engine.pitch_semitones == pytest.approx(2.0)

    lost = feed_empty(engine, range(800, 1201, 50))
    assert engine.pitch_semitones == pytest.approx(2.0)
    assert all(action.kind is not Kind.PITCH_RESET for result in lost for action in result.actions)
    assert any(result.engagement is not None and result.engagement.ended for result in lost)
    assert lost[-1].active_gesture is None

    high = with_alpha(1.0)
    high.step((make_hand("ILoveYou", 0.5, 0.5),), 0)
    high.step((make_hand("ILoveYou", 0.5, 0.5),), 150)
    for index in range(80):
        result = high.step((make_hand("ILoveYou", 0.5, 0.20),), 250 + index * 100)
        assert all(action.value is not None and action.value <= 12 for action in result.actions)
    assert high.pitch_semitones == pytest.approx(12.0)

    low = with_alpha(1.0)
    low.step((make_hand("ILoveYou", 0.5, 0.5),), 0)
    low.step((make_hand("ILoveYou", 0.5, 0.5),), 150)
    for index in range(80):
        result = low.step((make_hand("ILoveYou", 0.5, 0.80),), 250 + index * 100)
        assert all(action.value is not None and action.value >= -12 for action in result.actions)
    assert low.pitch_semitones == pytest.approx(-12.0)


def test_pitch_reset_on_still_hold() -> None:
    engine = with_alpha(1.0)
    end = _raise_pitch(engine, 0)
    assert engine.pitch_semitones != 0
    feed_empty(engine, range(end, end + 400, 50))
    assert engine.pitch_semitones != 0
    start = end + 400
    engine.step((make_hand("ILoveYou", 0.5, 0.5),), start)
    engine.step((make_hand("ILoveYou", 0.5, 0.5),), start + 150)
    reset = engine.step((make_hand("ILoveYou", 0.5, 0.5),), start + 150 + 1500)
    assert [action.kind for action in reset.actions] == [Kind.PITCH_RESET]
    assert reset.actions[0].value is None
    assert reset.actions[0].gesture == "ILoveYou"
    assert engine.pitch_semitones == pytest.approx(0.0)


def test_backwards_time_raises_and_toggles_report_state() -> None:
    engine = GestureEngine(default_config())
    engine.step((), 100)
    with pytest.raises(ValueError, match="backwards"):
        engine.step((), 50)
    assert engine.step((), 100).actions == ()
    assert engine.set_detection_enabled(False) is False
    assert engine.detection_enabled is False
    for t in range(150, 3001, 50):
        result = engine.step(_fists(t), t)
        assert result.actions == ()
        assert result.pause_toggled is False
    assert engine.paused is False
    assert engine.set_paused(True) is True
    assert engine.set_paused(True) is False
    assert engine.paused is True
    assert engine.set_paused(False) is True
    assert engine.pitch_semitones == pytest.approx(0.0)
    engine.set_detection_enabled(True)
    assert engine.detection_enabled is True


def _delta(gesture: str, x: float, y: float) -> float | None:
    engine = with_alpha(1.0)
    engine.step((make_hand(gesture, 0.5, 0.5),), 0)
    engine.step((make_hand(gesture, 0.5, 0.5),), 150)
    result = engine.step((make_hand(gesture, x, y),), 250)
    if not result.actions:
        return None
    return result.actions[0].value


def _raise_pitch(engine: GestureEngine, start: int) -> int:
    """Move ILoveYou up at full rate. Returns the timestamp after the last sample."""
    engine.step((make_hand("ILoveYou", 0.5, 0.5),), start)
    engine.step((make_hand("ILoveYou", 0.5, 0.5),), start + 150)
    last = start + 150
    for index in range(6):
        last = start + 250 + index * 100
        engine.step((make_hand("ILoveYou", 0.5, 0.20),), last)
    return last + 100


def _fists(t: int) -> tuple[Observation, Observation]:
    return (
        make_hand("Closed_Fist", 0.35, 0.5, handedness="Left", t=t),
        make_hand("Closed_Fist", 0.65, 0.5, handedness="Right", t=t),
    )
