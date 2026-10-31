"""Draw the preview overlay. The HUD never mutates engine state or the caller's frame."""

from __future__ import annotations

import cv2
import numpy as np

from gesture_media.hud_model import HudModel

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_ZONE_COLOR = (80, 220, 80)
_TEXT_COLOR = (255, 255, 255)
_SHADOW_COLOR = (0, 0, 0)
_BAR_COLOR = (40, 180, 255)
_BAR_OUTLINE = (190, 190, 190)
_LANDMARK_COLOR = (0, 220, 255)


def draw_hud(frame_bgr: np.ndarray, model: HudModel) -> np.ndarray:
    """Return a copy of `frame_bgr` with the overlay painted on it.

    When `model.hud_enabled` is false, the copy is unchanged.
    """
    canvas = frame_bgr.copy()
    if not model.hud_enabled:
        return canvas

    _draw_status(canvas, model)
    if model.show_landmarks:
        _draw_landmarks(canvas, model)
    _draw_zone(canvas, model)
    return canvas


def _draw_status(canvas: np.ndarray, model: HudModel) -> None:
    y = 22
    for line in _status_lines(model):
        _draw_text(canvas, line, 8, y)
        y += 18
    _draw_hold_bar(canvas, 8, y, model.hold_progress)


def _status_lines(model: HudModel) -> list[str]:
    if model.gesture is None:
        gesture_line = "gesture: -"
    elif model.score is None:
        gesture_line = f"gesture: {model.gesture}"
    else:
        gesture_line = f"gesture: {model.gesture} {model.score:.2f}"
    lines = [
        gesture_line,
        f"last: {model.last_action if model.last_action else '-'}",
        f"fps: {model.fps:.1f}",
        f"volume: {_volume_text(model.volume)}",
        f"mute: {_mute_text(model.muted)}",
        f"backend: {model.backend}",
    ]
    lines.extend(model.notices)
    if model.calibrated:
        lines.append("calibrated profile active")
    if model.paused:
        lines.append("paused")
    if not model.detection_enabled:
        lines.append("detection off")
    if model.vinyl_mode:
        lines.append("vinyl mode")
    return lines


def _volume_text(volume: float | None) -> str:
    if volume is None:
        return "-"
    return f"{volume:.0f}%"


def _mute_text(muted: bool | None) -> str:
    if muted is None:
        return "-"
    if muted:
        return "muted"
    return "unmuted"


def _draw_text(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(canvas, text, (x + 1, y + 1), _FONT, 0.5, _SHADOW_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), _FONT, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_hold_bar(canvas: np.ndarray, x: int, y: int, hold_progress: float) -> None:
    width = 160
    height = 12
    progress = min(1.0, max(0.0, float(hold_progress)))
    fill = int(progress * width)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), _BAR_OUTLINE, 1)
    if fill > 0:
        cv2.rectangle(canvas, (x, y), (x + fill, y + height), _BAR_COLOR, -1)


def _draw_landmarks(canvas: np.ndarray, model: HudModel) -> None:
    height, width = canvas.shape[:2]
    for hand in model.hands:
        for point in hand.landmarks:
            px = int(point.x * width)
            py = int(point.y * height)
            cv2.circle(canvas, (px, py), 3, _LANDMARK_COLOR, -1)


def _draw_zone(canvas: np.ndarray, model: HudModel) -> None:
    height, width = canvas.shape[:2]
    x_min, x_max, y_min, y_max = model.zone
    origin = (int(x_min * width), int(y_min * height))
    end = (int(x_max * width), int(y_max * height))
    cv2.rectangle(canvas, origin, end, _ZONE_COLOR, 2)
