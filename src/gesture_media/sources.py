"""Camera, video-file, and image-sequence frames behind one interface.

`read` returns BGR frames and does not mirror them. Mirror before recognition
so the on-screen right matches the user's right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class CameraOpenError(RuntimeError):
    """The requested camera could not be opened."""


class FrameSource(Protocol):
    """A sequence of BGR frames. Not mirrored."""

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None when this read fails or the source ends.

        A later `read` may overwrite the returned array.
        """

    def release(self) -> None:
        """Release the device or file. Safe to call more than once."""


def mirror_bgr(frame: np.ndarray) -> np.ndarray:
    """Flip a BGR frame left-to-right."""
    return cv2.flip(frame, 1)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Swap BGR channels to RGB for MediaPipe."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class _CaptureSource:
    """Shared read/release behavior for OpenCV captures."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture: cv2.VideoCapture | None = capture

    def read(self) -> np.ndarray | None:
        capture = self._capture
        if capture is None:
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()


class CameraSource(_CaptureSource):
    """Live webcam. One failed `read` returns None and does not raise."""

    def __init__(self, index: int, width: int, height: int, fps: float) -> None:
        capture = cv2.VideoCapture()
        if not capture.open(int(index)):
            capture.release()
            raise CameraOpenError(
                f"Could not open camera {index}. "
                "Check that the webcam is connected and not in use by another program."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        super().__init__(capture)


class VideoFileSource(_CaptureSource):
    """Frames from a video file. `read` returns None after the last frame."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        capture = cv2.VideoCapture()
        if not capture.open(str(self.path)):
            capture.release()
            raise RuntimeError(f"Could not open video file {self.path}")
        super().__init__(capture)


class ImageSequenceSource:
    """One sorted image file per `read`, then None.

    Names are sorted lexicographically. Non-image files are ignored.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.directory}")
        self._paths = _image_paths(self.directory)
        self._index = 0

    def read(self) -> np.ndarray | None:
        if self._index >= len(self._paths):
            return None
        path = self._paths[self._index]
        self._index += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return frame

    def release(self) -> None:
        """Nothing to release. Safe to call more than once."""


def open_source(config: object) -> FrameSource:
    """Open `config.source` when it is a file or directory, otherwise the camera.

    `config.camera` must have `index`, `width`, `height`, and `fps` when the
    camera is used. `config.source` is a path string, or None for the camera.
    """
    source = getattr(config, "source", None)
    if source:
        path = Path(source)
        if path.is_dir():
            return ImageSequenceSource(path)
        if path.is_file():
            return VideoFileSource(path)
    camera = getattr(config, "camera")
    return CameraSource(camera.index, camera.width, camera.height, camera.fps)


def _image_paths(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda path: path.name)
