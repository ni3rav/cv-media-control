"""Synthetic calibration samples, with no camera."""

from __future__ import annotations

import math

import pytest

from gesture_media.calibration import (
    CalibrationSamples,
    PalmSample,
    compute_calibration,
)
from gesture_media.config import Zone

DEFAULT_ZONE = Zone(0.15, 0.85, 0.10, 0.90)


def _point(x: float, y: float) -> tuple[PalmSample, ...]:
    return (PalmSample(x, y),)


def _steady_extents() -> dict[str, tuple[PalmSample, ...]]:
    return {
        "top": _point(0.5, 0.2),
        "bottom": _point(0.5, 0.8),
        "left": _point(0.2, 0.5),
        "right": _point(0.8, 0.5),
    }


def test_steady_hand_uses_floor_deadzone_and_reach() -> None:
    samples = CalibrationSamples(
        rest=tuple(PalmSample(0.5, 0.5) for _ in range(8)),
        **_steady_extents(),
    )
    result = compute_calibration(samples, min_score=0.6)
    assert result.deadzone == pytest.approx(0.03)
    assert result.max_disp == pytest.approx(0.24)
    assert result.zone.x_min == pytest.approx(0.15)
    assert result.zone.x_max == pytest.approx(0.85)
    assert result.zone.y_min == pytest.approx(0.15)
    assert result.zone.y_max == pytest.approx(0.85)
    assert result.warnings == ()


def test_jittery_hand_sets_deadzone_from_population_rms() -> None:
    rest = (
        PalmSample(0.40, 0.50),
        PalmSample(0.60, 0.50),
        PalmSample(0.50, 0.40),
        PalmSample(0.50, 0.60),
    )
    mean_x = sum(point.x for point in rest) / len(rest)
    mean_y = sum(point.y for point in rest) / len(rest)
    sigma = math.sqrt(
        sum((point.x - mean_x) ** 2 + (point.y - mean_y) ** 2 for point in rest) / len(rest)
    )
    samples = CalibrationSamples(
        rest=rest,
        top=_point(0.5, 0.0),
        bottom=_point(0.5, 1.0),
        left=_point(0.0, 0.5),
        right=_point(1.0, 0.5),
    )
    result = compute_calibration(samples, min_score=0.6)
    assert result.deadzone == pytest.approx(max(0.03, 3.0 * sigma))
    assert result.deadzone == pytest.approx(0.3)
    assert result.max_disp == pytest.approx(0.35)


def test_small_range_clamps_max_disp() -> None:
    samples = CalibrationSamples(
        rest=_point(0.5, 0.5) + _point(0.5, 0.5),
        top=_point(0.5, 0.48),
        bottom=_point(0.5, 0.52),
        left=_point(0.48, 0.5),
        right=_point(0.52, 0.5),
    )
    result = compute_calibration(samples, min_score=0.6)
    assert result.max_disp == pytest.approx(0.15)


def test_large_range_clamps_max_disp() -> None:
    samples = CalibrationSamples(
        rest=_point(0.5, 0.5) + _point(0.5, 0.5),
        top=_point(0.5, 0.0),
        bottom=_point(0.5, 1.0),
        left=_point(0.0, 0.5),
        right=_point(1.0, 0.5),
    )
    result = compute_calibration(samples, min_score=0.6)
    assert result.max_disp == pytest.approx(0.35)
    assert result.zone == Zone(0.0, 1.0, 0.0, 1.0)


def test_deadzone_is_lowered_when_jitter_exceeds_max_disp() -> None:
    samples = CalibrationSamples(
        rest=(PalmSample(0.0, 0.0), PalmSample(0.2, 0.0)),
        top=_point(0.1, 0.08),
        bottom=_point(0.1, 0.12),
        left=_point(0.08, 0.1),
        right=_point(0.12, 0.1),
    )
    result = compute_calibration(samples, min_score=0.6)
    assert result.max_disp == pytest.approx(0.15)
    assert result.deadzone == pytest.approx(0.13)
    assert result.deadzone < result.max_disp


def test_missing_samples_fall_back() -> None:
    empty = compute_calibration(CalibrationSamples(), min_score=0.6)
    assert empty.deadzone == pytest.approx(0.05)
    assert empty.max_disp == pytest.approx(0.25)
    assert empty.zone == DEFAULT_ZONE
    assert empty.warnings == ()

    one_rest = compute_calibration(CalibrationSamples(rest=_point(0.4, 0.4)), min_score=0.6)
    assert one_rest.deadzone == pytest.approx(0.05)
    assert one_rest.max_disp == pytest.approx(0.25)
    assert one_rest.zone == DEFAULT_ZONE

    vertical_only = compute_calibration(
        CalibrationSamples(
            rest=_point(0.5, 0.5) + _point(0.5, 0.5),
            top=_point(0.5, 0.2),
            bottom=_point(0.5, 0.8),
        ),
        min_score=0.6,
    )
    assert vertical_only.max_disp == pytest.approx(0.24)
    assert vertical_only.zone.x_min == pytest.approx(DEFAULT_ZONE.x_min)
    assert vertical_only.zone.x_max == pytest.approx(DEFAULT_ZONE.x_max)
    assert vertical_only.zone.y_min == pytest.approx(0.15)
    assert vertical_only.zone.y_max == pytest.approx(0.85)


def test_zone_clamps_to_the_frame() -> None:
    samples = CalibrationSamples(
        rest=_point(0.5, 0.5) + _point(0.5, 0.5),
        top=_point(0.5, 0.0),
        bottom=_point(0.5, 1.0),
        left=_point(0.0, 0.5),
        right=_point(1.0, 0.5),
    )
    zone = compute_calibration(samples, min_score=0.6).zone
    assert zone == Zone(0.0, 1.0, 0.0, 1.0)


def test_inverted_extents_use_the_default_zone() -> None:
    samples = CalibrationSamples(
        rest=_point(0.5, 0.5) + _point(0.5, 0.5),
        top=_point(0.5, 0.8),
        bottom=_point(0.5, 0.2),
        left=_point(0.8, 0.5),
        right=_point(0.2, 0.5),
    )
    result = compute_calibration(samples, min_score=0.6)
    assert result.zone == DEFAULT_ZONE


def test_low_gesture_scores_warn_without_changing_min_score() -> None:
    samples = CalibrationSamples(
        rest=_point(0.5, 0.5) + _point(0.5, 0.5),
        gesture_scores={
            "Open_Palm": (0.2, 0.4),
            "Victory": (0.5, 0.6),
            "Thumb_Up": (0.9, 0.8),
            "ILoveYou": (),
        },
    )
    result = compute_calibration(samples, min_score=0.6)
    assert len(result.warnings) == 2
    palm = next(warning for warning in result.warnings if warning.startswith("Open_Palm"))
    victory = next(warning for warning in result.warnings if warning.startswith("Victory"))
    assert "0.30" in palm
    assert "0.50" in palm
    assert "better lighting" in palm
    assert "0.55" in victory
    assert "Thumb_Up" not in " ".join(result.warnings)
    assert "ILoveYou" not in " ".join(result.warnings)
    assert not hasattr(result, "min_score")
