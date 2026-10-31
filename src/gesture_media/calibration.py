"""Pure calibration math. No camera and no file I/O."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from gesture_media.config import Zone

_DEFAULT_ZONE = Zone(0.15, 0.85, 0.10, 0.90)
_MARGIN = 0.05
_MIN_DEADZONE = 0.03
_DEFAULT_DEADZONE = 0.05
_DEFAULT_MAX_DISP = 0.25
_MIN_MAX_DISP = 0.15
_MAX_MAX_DISP = 0.35
_REACH_FRACTION = 0.8
_SUGGESTED_SCORE_FLOOR = 0.5


@dataclass(frozen=True, slots=True)
class PalmSample:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CalibrationSamples:
    rest: tuple[PalmSample, ...] = ()
    top: tuple[PalmSample, ...] = ()
    bottom: tuple[PalmSample, ...] = ()
    left: tuple[PalmSample, ...] = ()
    right: tuple[PalmSample, ...] = ()
    gesture_scores: Mapping[str, Sequence[float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CalibrationComputation:
    deadzone: float
    max_disp: float
    zone: Zone
    warnings: tuple[str, ...]


def compute_calibration(
    samples: CalibrationSamples,
    *,
    min_score: float,
) -> CalibrationComputation:
    """Turn recorded palm samples into deadzone, max displacement, and zone.

    Does not change ``min_score``. Gesture means below ``min_score`` produce
    warnings only.
    """
    deadzone = _deadzone(samples.rest)
    center = _center(samples)
    max_disp = _max_disp(center, samples)
    if deadzone >= max_disp:
        deadzone = max(_MIN_DEADZONE, max_disp - 0.02)
    return CalibrationComputation(
        deadzone=deadzone,
        max_disp=max_disp,
        zone=_zone(samples),
        warnings=tuple(_score_warnings(samples.gesture_scores, min_score)),
    )


def _deadzone(rest: Sequence[PalmSample]) -> float:
    if len(rest) < 2:
        return _DEFAULT_DEADZONE
    mean_x = sum(sample.x for sample in rest) / len(rest)
    mean_y = sum(sample.y for sample in rest) / len(rest)
    mean_square = sum((sample.x - mean_x) ** 2 + (sample.y - mean_y) ** 2 for sample in rest)
    sigma = math.sqrt(mean_square / len(rest))
    return max(_MIN_DEADZONE, 3.0 * sigma)


def _mean(samples: Sequence[PalmSample]) -> PalmSample | None:
    if not samples:
        return None
    count = len(samples)
    return PalmSample(
        sum(sample.x for sample in samples) / count,
        sum(sample.y for sample in samples) / count,
    )


def _center(samples: CalibrationSamples) -> PalmSample | None:
    if samples.rest:
        return _mean(samples.rest)
    extents = [
        point
        for point in (
            _mean(samples.top),
            _mean(samples.bottom),
            _mean(samples.left),
            _mean(samples.right),
        )
        if point is not None
    ]
    if not extents:
        return None
    count = len(extents)
    return PalmSample(
        sum(point.x for point in extents) / count,
        sum(point.y for point in extents) / count,
    )


def _max_disp(center: PalmSample | None, samples: CalibrationSamples) -> float:
    if center is None:
        return _DEFAULT_MAX_DISP
    top = _mean(samples.top)
    bottom = _mean(samples.bottom)
    left = _mean(samples.left)
    right = _mean(samples.right)
    vertical = _axis_reach(
        abs(center.y - top.y) if top is not None else None,
        abs(center.y - bottom.y) if bottom is not None else None,
    )
    horizontal = _axis_reach(
        abs(center.x - left.x) if left is not None else None,
        abs(center.x - right.x) if right is not None else None,
    )
    axes = [reach for reach in (vertical, horizontal) if reach is not None]
    if not axes:
        return _DEFAULT_MAX_DISP
    scaled = _REACH_FRACTION * min(axes)
    return min(_MAX_MAX_DISP, max(_MIN_MAX_DISP, scaled))


def _axis_reach(first: float | None, second: float | None) -> float | None:
    present = [reach for reach in (first, second) if reach is not None]
    if not present:
        return None
    return min(present)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _edge(point: float | None, fallback: float, margin: float) -> float:
    if point is None:
        return fallback
    return _clamp01(point + margin)


def _zone(samples: CalibrationSamples) -> Zone:
    left = _mean(samples.left)
    right = _mean(samples.right)
    top = _mean(samples.top)
    bottom = _mean(samples.bottom)
    zone = Zone(
        _edge(None if left is None else left.x, _DEFAULT_ZONE.x_min, -_MARGIN),
        _edge(None if right is None else right.x, _DEFAULT_ZONE.x_max, _MARGIN),
        _edge(None if top is None else top.y, _DEFAULT_ZONE.y_min, -_MARGIN),
        _edge(None if bottom is None else bottom.y, _DEFAULT_ZONE.y_max, _MARGIN),
    )
    if zone.x_min >= zone.x_max or zone.y_min >= zone.y_max:
        return _DEFAULT_ZONE
    return zone


def _score_warnings(
    gesture_scores: Mapping[str, Sequence[float]],
    min_score: float,
) -> list[str]:
    warnings: list[str] = []
    for gesture, scores in gesture_scores.items():
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        if mean >= min_score:
            continue
        suggested = max(_SUGGESTED_SCORE_FLOOR, mean)
        warnings.append(
            f"{gesture} average score {mean:.2f} is below min_score {min_score:.2f}; "
            f"use better lighting or lower min_score to {suggested:.2f}"
        )
    return warnings
