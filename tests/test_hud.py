"""HUD drawing on solid frames. No windows."""

from __future__ import annotations

import numpy as np

from gesture_media.hud import draw_hud
from gesture_media.hud_model import HudModel
from gesture_media.vision.types import Landmark, Observation


def test_draw_hud_disabled_matches_input() -> None:
    frame = np.full((48, 64, 3), 40, dtype=np.uint8)
    model = _model(
        hud_enabled=False,
        show_landmarks=True,
        hands=(_hand(),),
        zone=(0.0, 1.0, 0.0, 1.0),
        notices=("pitch unsupported on this backend",),
        calibrated=True,
        paused=True,
        detection_enabled=False,
        vinyl_mode=True,
    )

    drawn = draw_hud(frame, model)

    assert drawn is not frame
    assert np.array_equal(drawn, frame)


def test_status_lines_include_the_readout() -> None:
    from gesture_media.hud import _status_lines

    model = _model(
        gesture="Open_Palm",
        score=0.91,
        last_action="play_pause",
        fps=24.5,
        volume=40.0,
        muted=True,
        pitch_semitones=-1.5,
        backend="mpris",
        notices=("pitch unsupported on this backend", "player lost"),
        calibrated=True,
        paused=True,
        detection_enabled=False,
        vinyl_mode=True,
    )
    lines = _status_lines(model)
    assert lines[0] == "gesture: Open_Palm 0.91"
    assert "last: play_pause" in lines
    assert "fps: 24.5" in lines
    assert "volume: 40%" in lines
    assert "mute: muted" in lines
    assert "backend: mpris" in lines
    assert all(not line.startswith("pitch:") for line in lines)
    assert "pitch unsupported on this backend" in lines
    assert "player lost" in lines
    assert "calibrated profile active" in lines
    assert "paused" in lines
    assert "detection off" in lines
    assert "vinyl mode" in lines


def test_draw_hud_enabled_changes_zone_pixel() -> None:
    frame = np.full((480, 640, 3), (20, 30, 40), dtype=np.uint8)
    before = frame.copy()
    zone = (0.2, 0.8, 0.75, 0.95)
    model = _model(
        hud_enabled=True,
        gesture="Open_Palm",
        score=0.91,
        hold_progress=0.4,
        last_action="play_pause",
        fps=24.5,
        volume=40.0,
        muted=False,
        pitch_semitones=1.5,
        backend="mpris",
        notices=("pitch unsupported on this backend", "player lost"),
        zone=zone,
        show_landmarks=True,
        hands=(_hand(),),
        calibrated=True,
        paused=True,
        detection_enabled=False,
        vinyl_mode=True,
    )

    drawn = draw_hud(frame, model)

    assert np.array_equal(frame, before)
    assert not np.array_equal(drawn, frame)
    height, width = frame.shape[:2]
    x = int(zone[0] * width)
    y = int(zone[2] * height)
    assert not np.array_equal(drawn[y, x], frame[y, x])


def _hand() -> Observation:
    landmarks = tuple(Landmark(0.4, 0.5, 0.0) for _ in range(21))
    return Observation("Open_Palm", 0.91, "Right", landmarks, 0)


def _model(**overrides: object) -> HudModel:
    values: dict[str, object] = {
        "gesture": None,
        "score": None,
        "hold_progress": 0.0,
        "last_action": None,
        "fps": 0.0,
        "volume": None,
        "muted": None,
        "pitch_semitones": 0.0,
        "backend": "",
        "notices": (),
        "zone": (0.15, 0.85, 0.10, 0.90),
        "show_landmarks": False,
        "hands": (),
        "calibrated": False,
        "paused": False,
        "detection_enabled": True,
        "vinyl_mode": False,
        "hud_enabled": True,
    }
    values.update(overrides)
    return HudModel(**values)  # type: ignore[arg-type]
