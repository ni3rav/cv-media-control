#!/usr/bin/env python3
"""Download the MediaPipe gesture recognizer model into models/.

Skips the download when models/gesture_recognizer.task already exists and is
non-empty. The models directory is gitignored.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)


def destination_path() -> Path:
    """Repo-root models/gesture_recognizer.task, resolved from this file."""
    return Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"


def main() -> None:
    destination = destination_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(destination)
        return
    partial = destination.with_name(destination.name + ".partial")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(destination)


if __name__ == "__main__":
    main()
