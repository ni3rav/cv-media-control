"""Recognition results and the JSONL recording format.

Recordings store labels and landmarks only. Never images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PALM_INDEXES = (0, 5, 9, 13, 17)
LANDMARK_COUNT = 21


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class Observation:
    gesture: str
    score: float
    handedness: str
    landmarks: tuple[Landmark, ...]
    t_ms: int

    def __post_init__(self) -> None:
        if len(self.landmarks) != LANDMARK_COUNT:
            raise ValueError(f"expected {LANDMARK_COUNT} landmarks, got {len(self.landmarks)}")


def palm_center(observation: Observation) -> tuple[float, float, float]:
    """Mean of landmarks 0, 5, 9, 13, and 17."""
    chosen = [observation.landmarks[index] for index in PALM_INDEXES]
    count = len(chosen)
    return (
        sum(point.x for point in chosen) / count,
        sum(point.y for point in chosen) / count,
        sum(point.z for point in chosen) / count,
    )


def _landmark_from_seq(raw: object) -> Landmark:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("landmark must be a 3-item sequence")
    return Landmark(float(raw[0]), float(raw[1]), float(raw[2]))


def observation_to_dict(observation: Observation) -> dict[str, object]:
    return {
        "gesture": observation.gesture,
        "score": observation.score,
        "handedness": observation.handedness,
        "landmarks": [[point.x, point.y, point.z] for point in observation.landmarks],
    }


def observation_from_dict(data: dict[str, object], t_ms: int) -> Observation:
    landmarks_raw = data.get("landmarks")
    if not isinstance(landmarks_raw, list):
        raise ValueError("landmarks must be a list")
    return Observation(
        gesture=str(data["gesture"]),
        score=float(data["score"]),  # type: ignore[arg-type]
        handedness=str(data["handedness"]),
        landmarks=tuple(_landmark_from_seq(item) for item in landmarks_raw),
        t_ms=t_ms,
    )


def frame_to_json(t_ms: int, hands: tuple[Observation, ...]) -> str:
    payload = {
        "t_ms": t_ms,
        "hands": [observation_to_dict(hand) for hand in hands],
    }
    return json.dumps(payload, separators=(",", ":"))


def frame_from_json(line: str) -> tuple[int, tuple[Observation, ...]]:
    data = json.loads(line)
    if not isinstance(data, dict) or "t_ms" not in data or "hands" not in data:
        raise ValueError("recording line must contain t_ms and hands")
    t_ms = int(data["t_ms"])
    hands_raw = data["hands"]
    if not isinstance(hands_raw, list):
        raise ValueError("hands must be a list")
    hands = tuple(observation_from_dict(item, t_ms) for item in hands_raw)
    return t_ms, hands
