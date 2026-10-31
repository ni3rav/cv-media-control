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
    aim: str | None = None,
) -> Observation:
    landmarks = [
        Landmark(x, y, 0.0) if index in PALM else Landmark(0.0, 0.0, 0.0) for index in range(21)
    ]
    if aim is not None:
        dx, dy = {
            "right": (0.12, 0.0),
            "left": (-0.12, 0.0),
            "up": (0.0, -0.12),
            "down": (0.0, 0.12),
        }[aim]
        landmarks[2] = Landmark(x, y, 0.0)
        landmarks[4] = Landmark(x + dx, y + dy, 0.0)
    return Observation(gesture, score, handedness, tuple(landmarks), t)


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
        ("Closed_Fist", Kind.PLAY_PAUSE, 350),
        ("Open_Palm", Kind.MUTE_TOGGLE, 350),
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
            assert result.hold_progress == pytest.approx(1.0)
    assert fired_at is not None
    assert fire_at <= fired_at < fire_at + 50
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
    feed(engine, "Closed_Fist", 0.5, 0.5, range(0, 201, 50))
    assert kinds(feed_empty(engine, range(250, 601, 50))) == []
    results = feed(engine, "Closed_Fist", 0.5, 0.5, range(650, 1401, 50))
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
    assert kinds(fired) == [Kind.MUTE_TOGGLE]


def test_grace_period_keeps_hold_progress() -> None:
    engine = GestureEngine(default_config())
    feed(engine, "Closed_Fist", 0.5, 0.5, range(0, 351, 50))
    for t in (400, 450):
        missing = engine.step((), t)
        assert missing.active_gesture == "Closed_Fist"
        assert missing.actions == ()
    fired = engine.step((make_hand("Closed_Fist", 0.5, 0.5),), 500)
    assert [action.kind for action in fired.actions] == [Kind.PLAY_PAUSE]
    assert fired.hold_progress == pytest.approx(1.0)


def test_lock_until_release_then_cooldown() -> None:
    engine = GestureEngine(default_config())
    held = feed(engine, "Closed_Fist", 0.5, 0.5, range(0, 1501, 50))
    assert kinds(held) == [Kind.PLAY_PAUSE]
    assert held[-1].hold_progress == pytest.approx(1.0)
    released_at = None
    gaps = range(1550, 2001, 50)
    for result, t in zip(feed_empty(engine, gaps), gaps, strict=True):
        if result.active_gesture is None and released_at is None:
            released_at = t
    assert released_at == 1800
    during = feed(engine, "Closed_Fist", 0.5, 0.5, range(2050, 2500, 50))
    assert kinds(during) == []
    assert all(result.active_gesture is None for result in during)
    after = feed(engine, "Closed_Fist", 0.5, 0.5, range(2500, 3601, 50))
    assert kinds(after) == [Kind.PLAY_PAUSE]


def test_hand_loss_clears_locks_but_not_cooldowns() -> None:
    fresh = GestureEngine(default_config())
    feed(fresh, "Closed_Fist", 0.5, 0.5, range(0, 801, 50))
    release = feed_empty(fresh, range(850, 1101, 50))
    assert release[-1].active_gesture is None
    blocked = feed(fresh, "Closed_Fist", 0.5, 0.5, range(1150, 1801, 50))
    assert kinds(blocked) == []
    assert all(result.active_gesture is None for result in blocked)
    thumb = feed(fresh, "Thumb_Up", 0.5, 0.5, range(1850, 2501, 50))
    assert Kind.VOLUME in kinds(thumb)
    assert any(
        result.active_gesture == "Thumb_Up" and t < 2100 for t, result in _with_times(thumb, 1850)
    )


def _with_times(results: list[StepResult], start: int) -> list[tuple[int, StepResult]]:
    return [(start + index * 50, result) for index, result in enumerate(results)]


def test_continuous_deadzone_clamp_and_directions() -> None:
    # alpha 1 makes one tick equal the raw displacement.
    # One finger seeks sideways. Up and down does not change pitch.
    assert _delta("Pointing_Up", 0.54, 0.5) is None  # inside the deadzone
    assert _delta("Pointing_Up", 0.56, 0.5) == pytest.approx(0.1)
    assert _delta("Pointing_Up", 0.80, 0.5) == pytest.approx(2.0)
    assert _delta("Pointing_Up", 0.20, 0.5) == pytest.approx(-2.0)
    assert _delta("Pointing_Up", 0.5, 0.20) is None
    assert _delta("Pointing_Up", 0.5, 0.80) is None
    assert _delta("Thumb_Up", 0.5, 0.5) == pytest.approx(2.0)
    assert _delta("Thumb_Down", 0.5, 0.5) == pytest.approx(-2.0)


def test_default_smoothing_ramps_toward_the_clamp() -> None:
    engine = GestureEngine(default_config())
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 0)
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 150)
    values = []
    for index in range(20):
        result = engine.step((make_hand("Pointing_Up", 0.75, 0.5),), 250 + index * 100)
        values.extend(action.value for action in result.actions if action.value is not None)
    assert values[0] == pytest.approx(0.5)
    assert values[-1] == pytest.approx(2.0, abs=0.05)
    assert all(abs(value) <= 2.0 + 1e-9 for value in values)


def test_pointing_held_still_does_not_fire() -> None:
    engine = GestureEngine(default_config())
    held = feed(engine, "Pointing_Up", 0.5, 0.5, range(0, 2001, 50))
    assert kinds(held) == []
    assert all(action.kind is not Kind.PITCH for result in held for action in result.actions)


def test_hold_still_is_suppressed_after_leaving_the_deadzone() -> None:
    engine = GestureEngine(default_config())
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 0)
    engine.step((make_hand("Pointing_Up", 0.5, 0.5),), 150)
    moved = engine.step((make_hand("Pointing_Up", 0.5, 0.30),), 250)
    assert moved.engagement is not None
    assert moved.hold_progress == pytest.approx(0.0)
    for t in (350, 1850, 2500, 4000):
        result = engine.step((make_hand("Pointing_Up", 0.5, 0.5),), t)
        assert all(action.kind is not Kind.PITCH_RESET for action in result.actions)
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
    assert kinds(feed(boundary, "Open_Palm", 0.15, 0.5, range(0, 801, 50))) == [Kind.MUTE_TOGGLE]
    exact = GestureEngine(default_config())
    assert set(kinds(feed(exact, "Thumb_Up", 0.5, 0.5, range(0, 801, 50), score=0.6))) == {
        Kind.VOLUME
    }


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
    assert kinds(fired)
    assert set(kinds(fired)) == {Kind.VOLUME}


def test_two_fist_pause_and_suppression() -> None:
    engine = GestureEngine(default_config())
    toggles = []
    for t in range(0, 2501, 50):
        result = engine.step(_fists(t), t)
        if result.pause_toggled:
            toggles.append((t, result.paused))
        assert result.actions == ()
        if t == 700:
            assert result.active_gesture == "two_fists"
            assert result.hold_progress == pytest.approx(0.5)
    assert toggles == [(1400, True)]
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
    assert resume == [(6850, False)]
    assert engine.paused is False
    feed_empty(engine, range(8050, 8401, 50))
    played = feed(engine, "Closed_Fist", 0.5, 0.5, range(8450, 9501, 50))
    assert kinds(played) == [Kind.PLAY_PAUSE]


def test_two_fist_hold_too_short_does_nothing() -> None:
    engine = GestureEngine(default_config())
    for t in range(0, 1400, 50):
        result = engine.step(_fists(t), t)
        assert result.pause_toggled is False
    trailing = feed_empty(engine, range(1600, 4001, 50))
    assert all(not result.pause_toggled for result in trailing)
    assert engine.paused is False


def test_single_fist_plays_and_two_hands_suppress_single_hand_actions() -> None:
    single = GestureEngine(default_config())
    assert kinds(feed(single, "Closed_Fist", 0.5, 0.5, range(0, 801, 50))) == [Kind.PLAY_PAUSE]
    palm = GestureEngine(default_config())
    assert kinds(feed(palm, "Open_Palm", 0.5, 0.5, range(0, 801, 50))) == [Kind.MUTE_TOGGLE]

    outside = GestureEngine(default_config())
    actions = []
    for t in range(0, 801, 50):
        hands = (
            make_hand("Closed_Fist", 0.5, 0.5, t=t),
            make_hand("Closed_Fist", 0.02, 0.5, t=t),
        )
        actions.extend(action.kind for action in outside.step(hands, t).actions)
    assert actions == [Kind.PLAY_PAUSE]
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
    right = feed(filtered, "Closed_Fist", 0.5, 0.5, range(0, 1001, 50), handedness="Right")
    assert kinds(right) == []
    left = feed(filtered, "Closed_Fist", 0.5, 0.5, range(1100, 2101, 50), handedness="Left")
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


@pytest.mark.parametrize("gesture", ["Thumb_Up", "Thumb_Down"])
def test_sideways_thumb_changes_track(gesture: str) -> None:
    right = GestureEngine(default_config())
    assert kinds(feed(right, gesture, 0.5, 0.5, range(0, 801, 50), aim="right")) == [Kind.NEXT]
    left = GestureEngine(default_config())
    assert kinds(feed(left, gesture, 0.5, 0.5, range(0, 801, 50), aim="left")) == [Kind.PREVIOUS]


def _delta(gesture: str, x: float, y: float) -> float | None:
    engine = with_alpha(1.0)
    engine.step((make_hand(gesture, 0.5, 0.5),), 0)
    engine.step((make_hand(gesture, 0.5, 0.5),), 150)
    result = engine.step((make_hand(gesture, x, y),), 250)
    if not result.actions:
        return None
    return result.actions[0].value


def _fists(t: int) -> tuple[Observation, Observation]:
    return (
        make_hand("Closed_Fist", 0.35, 0.5, handedness="Left", t=t),
        make_hand("Closed_Fist", 0.65, 0.5, handedness="Right", t=t),
    )
