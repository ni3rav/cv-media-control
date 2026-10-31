"""Frame sources, without opening a real webcam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gesture_media.sources import (
    CameraOpenError,
    CameraSource,
    ImageSequenceSource,
    VideoFileSource,
    bgr_to_rgb,
    mirror_bgr,
    open_source,
)


def test_mirror_bgr_flips_left_and_right() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    frame[1, 0] = (4, 5, 6)

    mirrored = mirror_bgr(frame)

    assert tuple(mirrored[0, -1]) == (1, 2, 3)
    assert tuple(mirrored[1, -1]) == (4, 5, 6)
    assert tuple(int(v) for v in mirrored[0, 0]) == (0, 0, 0)
    assert tuple(frame[0, 0]) == (1, 2, 3)


def test_bgr_to_rgb_swaps_channels() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30

    rgb = bgr_to_rgb(frame)

    assert int(rgb[0, 0, 0]) == 30
    assert int(rgb[0, 0, 1]) == 20
    assert int(rgb[0, 0, 2]) == 10
    assert int(frame[0, 0, 0]) == 10


def test_image_sequence_yields_sorted_files_then_none(tmp_path: Path) -> None:
    _write_png(tmp_path, "b.png", 20)
    _write_png(tmp_path, "c.png", 30)
    _write_png(tmp_path, "a.png", 10)
    (tmp_path / "000.txt").write_text("not an image")

    source = open_source(SimpleNamespace(source=str(tmp_path), camera=None))
    assert isinstance(source, ImageSequenceSource)

    first = source.read()
    second = source.read()
    third = source.read()
    assert first is not None and int(first[0, 0, 0]) == 10
    assert second is not None and int(second[0, 0, 0]) == 20
    assert third is not None and int(third[0, 0, 0]) == 30
    assert source.read() is None
    source.release()
    source.release()


def test_video_file_source_reads_until_end(tmp_path: Path) -> None:
    path = tmp_path / "tiny.avi"
    _write_tiny_avi(path)

    source = open_source(SimpleNamespace(source=str(path), camera=None))
    assert isinstance(source, VideoFileSource)
    frames = []
    for _ in range(4):
        frame = source.read()
        if frame is None:
            break
        frames.append(frame)
    assert len(frames) == 2
    assert frames[0].shape == (16, 16, 3)
    assert source.read() is None
    source.release()
    source.release()
    assert source.read() is None


def test_camera_open_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeCapture:
        def open(self, index: int) -> bool:
            self.index = index
            return False

        def release(self) -> None:
            self.released = True

    def factory(*_args: object, **_kwargs: object) -> FakeCapture:
        capture = FakeCapture()
        created["capture"] = capture
        return capture

    monkeypatch.setattr("gesture_media.sources.cv2.VideoCapture", factory)

    with pytest.raises(CameraOpenError, match="camera 3"):
        CameraSource(3, 640, 480, 30)
    assert created["capture"].released is True  # type: ignore[attr-defined]


def test_camera_read_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self) -> None:
            self.sets: list[tuple[int, float]] = []
            self.releases = 0
            self.reads = 0

        def open(self, index: int) -> bool:
            self.index = index
            return True

        def set(self, prop: int, value: float) -> bool:
            self.sets.append((prop, value))
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.reads += 1
            if self.reads == 1:
                return True, frame
            return False, None

        def release(self) -> None:
            self.releases += 1

    holder: dict[str, FakeCapture] = {}

    def factory(*_args: object, **_kwargs: object) -> FakeCapture:
        capture = FakeCapture()
        holder["capture"] = capture
        return capture

    monkeypatch.setattr("gesture_media.sources.cv2.VideoCapture", factory)
    camera = SimpleNamespace(index=1, width=320, height=240, fps=15)
    source = open_source(SimpleNamespace(source=None, camera=camera))

    assert isinstance(source, CameraSource)
    assert holder["capture"].index == 1
    props = dict(holder["capture"].sets)
    assert props[cv2.CAP_PROP_FRAME_WIDTH] == 320
    assert props[cv2.CAP_PROP_FRAME_HEIGHT] == 240
    assert props[cv2.CAP_PROP_FPS] == 15
    assert source.read() is frame
    assert source.read() is None
    source.release()
    source.release()
    assert holder["capture"].releases == 1
    assert source.read() is None


def test_missing_source_path_uses_camera(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, tuple[object, ...]] = {}

    class FakeCamera:
        def __init__(self, index: int, width: int, height: int, fps: float) -> None:
            seen["args"] = (index, width, height, fps)

        def read(self) -> None:
            return None

        def release(self) -> None:
            return None

    monkeypatch.setattr("gesture_media.sources.CameraSource", FakeCamera)
    camera = SimpleNamespace(index=4, width=640, height=480, fps=30)
    source = open_source(SimpleNamespace(source=str(tmp_path / "missing.mp4"), camera=camera))

    assert isinstance(source, FakeCamera)
    assert seen["args"] == (4, 640, 480, 30)


def _write_png(directory: Path, name: str, value: int) -> None:
    image = np.full((6, 6, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(directory / name), image)


def _write_tiny_avi(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV VideoWriter could not open an MJPG avi")
    first = np.zeros((16, 16, 3), dtype=np.uint8)
    second = np.zeros((16, 16, 3), dtype=np.uint8)
    second[:, :] = (0, 255, 0)
    writer.write(first)
    writer.write(second)
    writer.release()
