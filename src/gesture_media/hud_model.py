"""Immutable snapshot the preview draws. The HUD never mutates engine state."""

from __future__ import annotations

from dataclasses import dataclass

from gesture_media.vision.types import Observation


@dataclass(frozen=True, slots=True)
class HudModel:
    gesture: str | None
    score: float | None
    hold_progress: float
    last_action: str | None
    fps: float
    volume: float | None
    muted: bool | None
    pitch_semitones: float
    backend: str
    notices: tuple[str, ...]
    zone: tuple[float, float, float, float]
    show_landmarks: bool
    hands: tuple[Observation, ...]
    calibrated: bool
    paused: bool
    detection_enabled: bool
    vinyl_mode: bool
    hud_enabled: bool
