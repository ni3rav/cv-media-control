"""Classifier interface. Implementations live in classifier.py and rules.py."""

from __future__ import annotations

from typing import Protocol

from gesture_media.vision.types import Observation


class GestureClassifier(Protocol):
    """Turn one RGB frame into zero, one, or two hand observations.

    `timestamp_ms` must be strictly increasing across calls for VIDEO mode.
    An empty tuple means no hand. The frame is already mirrored.
    """

    def recognize(self, rgb_frame: object, timestamp_ms: int) -> tuple[Observation, ...]:
        """Classify `rgb_frame` (a numpy ndarray, HxWx3, RGB)."""

    def close(self) -> None:
        """Release native resources."""
