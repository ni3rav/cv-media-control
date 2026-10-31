"""Rule stub, plus MediaPipe mapping that skips without the model file."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gesture_media.vision.rules import (
    RulesClassifier,
    classify_landmarks,
    landmarks_to_observation,
)
from gesture_media.vision.types import Landmark


def test_classify_landmarks_returns_none() -> None:
    landmarks = tuple(Landmark(0.0, 0.0, 0.0) for _ in range(21))
    assert classify_landmarks(()) is None
    assert classify_landmarks(landmarks) is None
    assert landmarks_to_observation(landmarks, 12) is None


def test_rules_classifier_recognize_returns_empty() -> None:
    classifier = RulesClassifier()
    assert classifier.recognize(object(), 0) == ()
    classifier.close()


def test_landmarks_to_observation_uses_classify_landmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    landmarks = tuple(Landmark(0.2, 0.3, 0.4) for _ in range(21))
    monkeypatch.setattr(
        "gesture_media.vision.rules.classify_landmarks",
        lambda _landmarks: "Victory",
    )

    observation = landmarks_to_observation(landmarks, 40)

    assert observation is not None
    assert observation.gesture == "Victory"
    assert observation.score == 1.0
    assert observation.t_ms == 40
    assert observation.landmarks == landmarks
    assert RulesClassifier().recognize(object(), 40) == ()


@pytest.fixture
def classifier_module():
    pytest.importorskip("mediapipe")
    import gesture_media.vision.classifier as classifier

    return classifier


def test_monotonic_timestamp(classifier_module) -> None:
    fn = classifier_module.monotonic_timestamp
    assert fn(0, None) == 0
    assert fn(4, 4) == 5
    assert fn(3, 9) == 10
    assert fn(12, 9) == 12


def test_observations_from_result_maps_hands(classifier_module) -> None:
    empty = SimpleNamespace(hand_landmarks=[], gestures=[], handedness=[])
    assert classifier_module.observations_from_result(empty, 7, 2) == ()

    points = [SimpleNamespace(x=index / 21, y=0.2, z=-0.1) for index in range(21)]
    top = SimpleNamespace(category_name="Thumb_Up", score=0.8, display_name=None)
    hand = SimpleNamespace(category_name="Left", score=0.99, display_name=None)
    result = SimpleNamespace(
        hand_landmarks=[points, points, points],
        gestures=[[top], [], [top]],
        handedness=[[hand], [], []],
    )

    observations = classifier_module.observations_from_result(result, 11, 2)

    assert len(observations) == 2
    assert observations[0].gesture == "Thumb_Up"
    assert observations[0].score == 0.8
    assert observations[0].handedness == "Left"
    assert observations[0].t_ms == 11
    assert len(observations[0].landmarks) == 21
    assert observations[0].landmarks[1].x == pytest.approx(1 / 21)
    assert observations[1].gesture == "None"
    assert observations[1].score == 0.0
    assert observations[1].handedness == ""


def test_observations_from_result_skips_short_hands(classifier_module) -> None:
    top = SimpleNamespace(category_name="Open_Palm", score=0.5, display_name=None)
    result = SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=0.0, y=0.0, z=0.0)]],
        gestures=[[top]],
        handedness=[[SimpleNamespace(category_name="Right", score=1.0, display_name=None)]],
    )
    assert classifier_module.observations_from_result(result, 1, 2) == ()


def test_mediapipe_classifier_skips_without_model(classifier_module) -> None:
    model = Path(__file__).resolve().parents[1] / "models" / "gesture_recognizer.task"
    if not model.is_file() or model.stat().st_size == 0:
        pytest.skip("models/gesture_recognizer.task is absent")

    import numpy as np

    classifier = classifier_module.MediaPipeClassifier(model, num_hands=2, min_score=0.6)
    assert classifier.min_score == 0.6
    assert classifier.num_hands == 2
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    stamps: list[int] = []
    original = classifier._recognizer.recognize_for_video

    def wrapped(image: object, timestamp_ms: int):
        stamps.append(timestamp_ms)
        return original(image, timestamp_ms)

    classifier._recognizer.recognize_for_video = wrapped
    try:
        first = classifier.recognize(frame, 0)
        second = classifier.recognize(frame, 0)
        third = classifier.recognize(frame, 15)
    finally:
        classifier.close()
        classifier.close()

    assert stamps == [0, 1, 15]
    for batch, stamp in ((first, 0), (second, 1), (third, 15)):
        assert isinstance(batch, tuple)
        assert len(batch) <= 2
        for observation in batch:
            assert observation.t_ms == stamp
            assert len(observation.landmarks) == 21
