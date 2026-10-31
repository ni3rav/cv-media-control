"""Palm center and recording JSON roundtrip."""

from __future__ import annotations

import pytest

from gesture_media.vision.types import (
    LANDMARK_COUNT,
    Landmark,
    Observation,
    frame_from_json,
    frame_to_json,
    palm_center,
)


def _observation() -> Observation:
    landmarks = []
    for index in range(LANDMARK_COUNT):
        if index == 0:
            landmarks.append(Landmark(1.0, 2.0, 3.0))
        elif index == 5:
            landmarks.append(Landmark(2.0, 3.0, 4.0))
        elif index == 9:
            landmarks.append(Landmark(3.0, 4.0, 5.0))
        elif index == 13:
            landmarks.append(Landmark(4.0, 5.0, 6.0))
        elif index == 17:
            landmarks.append(Landmark(5.0, 6.0, 7.0))
        else:
            landmarks.append(Landmark(9.0, 9.0, 9.0))
    return Observation("Open_Palm", 0.8, "Right", tuple(landmarks), 40)


def test_palm_center_is_mean_of_wrist_and_knuckles() -> None:
    center = palm_center(_observation())
    assert center == pytest.approx((3.0, 4.0, 5.0))


def test_frame_json_roundtrip() -> None:
    first = _observation()
    other_points = tuple(Landmark(float(i), float(i) / 10.0, 0.25) for i in range(LANDMARK_COUNT))
    second = Observation("Closed_Fist", 0.91, "Left", other_points, 40)
    line = frame_to_json(40, (first, second))
    t_ms, hands = frame_from_json(line)
    assert t_ms == 40
    assert len(hands) == 2
    assert hands[0].gesture == "Open_Palm"
    assert hands[0].score == pytest.approx(0.8)
    assert hands[0].handedness == "Right"
    assert hands[0].t_ms == 40
    assert hands[0].landmarks == first.landmarks
    assert hands[1].gesture == "Closed_Fist"
    assert hands[1].landmarks == other_points
    assert palm_center(hands[0]) == pytest.approx((3.0, 4.0, 5.0))


def test_frame_json_rejects_bad_payloads() -> None:
    with pytest.raises(ValueError):
        frame_from_json('{"hands": []}')
    with pytest.raises(ValueError):
        Observation("Open_Palm", 1.0, "Right", (), 0)
