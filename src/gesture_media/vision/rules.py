"""Stub landmark-rule classifier.

The author replaces `classify_landmarks`. The engine keeps the same labels
when this classifier is swapped in for MediaPipe.
"""

from __future__ import annotations

from gesture_media.vision.types import Observation

# MediaPipe built-in labels this stub is meant to produce.
GESTURE_LABELS = (
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
)


def classify_landmarks(landmarks: tuple) -> str | None:
    """Extension point. Return a MediaPipe label or None. The author replaces this."""
    return None


def landmarks_to_observation(landmarks: tuple, t_ms: int) -> Observation | None:
    """Build an observation when `classify_landmarks` returns a label.

    The score is 1.0 and handedness is empty until the author fills those in.
    Returns None when the extension point has no label yet.
    """
    label = classify_landmarks(landmarks)
    if label is None:
        return None
    return Observation(
        gesture=label,
        score=1.0,
        handedness="",
        landmarks=tuple(landmarks),
        t_ms=t_ms,
    )


class RulesClassifier:
    """Stub GestureClassifier. `recognize` returns no hands until the author fills it in.

    Landmark rules should emit these MediaPipe labels:
    Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou.
    """

    def recognize(self, rgb_frame: object, timestamp_ms: int) -> tuple[Observation, ...]:
        """Return no hands. Use `landmarks_to_observation` once rules exist."""
        return ()

    def close(self) -> None:
        """Nothing to release."""
