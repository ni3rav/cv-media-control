"""MediaPipe GestureRecognizer in VIDEO mode.

`min_score` is stored for callers. The engine decides which scores to keep.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np

from gesture_media.vision.types import LANDMARK_COUNT, Landmark, Observation


def monotonic_timestamp(timestamp_ms: int, last_timestamp_ms: int | None) -> int:
    """Return a timestamp strictly greater than the previous VIDEO-mode call."""
    if last_timestamp_ms is not None and timestamp_ms <= last_timestamp_ms:
        return last_timestamp_ms + 1
    return int(timestamp_ms)


def observations_from_result(result, t_ms: int, num_hands: int) -> tuple[Observation, ...]:
    """Map a GestureRecognizer result to one observation per detected hand."""
    observations: list[Observation] = []
    count = min(len(result.hand_landmarks), num_hands)
    for index in range(count):
        categories = result.gestures[index] if index < len(result.gestures) else ()
        handedness = result.handedness[index] if index < len(result.handedness) else ()
        observation = _observation_for_hand(
            result.hand_landmarks[index],
            categories,
            handedness,
            t_ms,
        )
        if observation is not None:
            observations.append(observation)
    return tuple(observations)


class MediaPipeClassifier:
    """Recognize up to `num_hands` hands. An empty tuple means no hand."""

    def __init__(self, model_path: Path, *, num_hands: int = 2, min_score: float = 0.6) -> None:
        self.model_path = Path(model_path)
        self.num_hands = num_hands
        self.min_score = min_score
        self._last_timestamp_ms: int | None = None
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def recognize(self, rgb_frame: object, timestamp_ms: int) -> tuple[Observation, ...]:
        """Classify one mirrored RGB uint8 frame (HxWx3)."""
        timestamp = monotonic_timestamp(timestamp_ms, self._last_timestamp_ms)
        self._last_timestamp_ms = timestamp
        frame = np.ascontiguousarray(rgb_frame)
        if not frame.flags.writeable:
            frame = frame.copy()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._recognizer.recognize_for_video(image, timestamp)
        return observations_from_result(result, timestamp, self.num_hands)

    def close(self) -> None:
        """Release the native recognizer. Safe to call more than once."""
        recognizer = getattr(self, "_recognizer", None)
        self._recognizer = None
        if recognizer is not None:
            recognizer.close()


def _category_name(category: object) -> str:
    name = getattr(category, "category_name", None) or getattr(category, "display_name", None)
    return str(name) if name else ""


def _observation_for_hand(points, categories, handedness, t_ms: int) -> Observation | None:
    if len(points) != LANDMARK_COUNT:
        return None
    landmarks = tuple(
        Landmark(float(point.x), float(point.y), float(0.0 if point.z is None else point.z))
        for point in points
    )
    gesture = "None"
    score = 0.0
    if categories:
        top = categories[0]
        gesture = _category_name(top) or "None"
        raw_score = getattr(top, "score", None)
        if raw_score is not None:
            score = float(raw_score)
    hand_name = _category_name(handedness[0]) if handedness else ""
    return Observation(gesture, score, hand_name, landmarks, t_ms)
