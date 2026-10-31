"""mpv backend. The live libmpv test is skipped when the library is missing."""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from gesture_media.players.base import pitch_scale
from gesture_media.players.mpv import MpvPlayer, MpvUnavailable, rubberband_available


def _libmpv() -> bool:
    try:
        import mpv  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


class _FakeMpv:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.speed = 1.0
        self.pause = False
        self.volume = 30.0
        self.mute = False
        self.force_window = kwargs.get("force_window")
        self.af = None
        self.commands: list[tuple] = []
        self.loaded: list[tuple[str, str]] = []
        self.terminated = False
        self.time_pos = 1.5
        _FakeMpv.latest = self

    def command(self, *args) -> None:
        self.commands.append(args)

    def loadfile(self, path, mode="replace") -> None:
        self.loaded.append((path, mode))

    def terminate(self) -> None:
        self.terminated = True


class _FakeModule:
    MPV = _FakeMpv


def _install_fake(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mpv", _FakeModule())


def test_constructor_raises_when_libmpv_is_missing():
    if MpvPlayer.available():
        pytest.skip("libmpv is installed")
    with pytest.raises(MpvUnavailable):
        MpvPlayer(())


def test_options_pitch_and_playlist(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr("gesture_media.players.mpv.rubberband_available", lambda: True)
    player = MpvPlayer(("a.mp3", "b.mp3"), force_window=False, ao="null", vo="null")
    core = _FakeMpv.latest
    assert core.kwargs["input_default_bindings"] is False
    assert core.kwargs["input_vo_keyboard"] is False
    assert core.kwargs["osc"] is True
    assert core.kwargs["ytdl"] is False
    assert core.kwargs["idle"] is True
    assert core.kwargs["force_window"] is False
    assert core.kwargs["ao"] == "null"
    assert core.kwargs["vo"] == "null"
    assert core.loaded == [("a.mp3", "replace"), ("b.mp3", "append")]
    player.set_pitch(2)
    scale = f"{pitch_scale(2):.6f}"
    assert ("af-command", "rb", "set-pitch", scale) in core.commands
    assert core.af == "@rb:rubberband=pitch-scale=1"
    assert core.speed == 1
    assert player.pitch_semitones == 2
    assert player.vinyl_mode is False
    assert player.is_available() is True
    player.play_pause()
    assert core.pause is True
    player.next()
    player.previous()
    player.seek(4)
    assert ("playlist-next",) in core.commands
    assert ("playlist-prev",) in core.commands
    assert ("seek", 4.0, "relative") in core.commands
    player.volume_delta(80)
    assert core.volume == 100
    player.toggle_mute()
    assert player.get_state() == (100.0, True)
    assert player.get_position() == 1.5
    player.open()
    assert core.force_window is True
    player.close()
    assert player.is_available() is False
    player.close()
    assert core.terminated is True


def test_volume_up_clears_mute_and_string_no_is_not_muted(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr("gesture_media.players.mpv.rubberband_available", lambda: True)
    player = MpvPlayer(())
    core = _FakeMpv.latest
    core.mute = "no"
    assert player.get_state() == (30.0, False)
    player.toggle_mute()
    assert core.mute is True
    player.volume_delta(1)
    assert core.mute is False


def test_vinyl_mode_changes_speed(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr("gesture_media.players.mpv.rubberband_available", lambda: False)
    player = MpvPlayer(())
    player.set_pitch(2)
    assert player.vinyl_mode is True
    assert player.pitch_semitones == 2
    assert _FakeMpv.latest.speed == pytest.approx(pitch_scale(2))


def test_mpv_errors_are_swallowed(monkeypatch):
    class Boom(_FakeMpv):
        def command(self, *args) -> None:
            raise RuntimeError("filter graph")

    module = _FakeModule()
    module.MPV = Boom
    monkeypatch.setitem(sys.modules, "mpv", module)
    monkeypatch.setattr("gesture_media.players.mpv.rubberband_available", lambda: True)
    player = MpvPlayer(())
    player.seek(1)
    player.set_pitch(1)
    player.close()


def test_rubberband_probe_is_cached(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs["timeout"] == 2.0
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="rubberband Pitch conversion\n",
            stderr="",
        )

    monkeypatch.setattr("gesture_media.players.mpv.subprocess.run", fake_run)
    previous = rubberband_available.cache
    rubberband_available.cache = None
    try:
        assert rubberband_available() is True
        assert rubberband_available() is True
        assert calls == [["mpv", "--af=help"]]
    finally:
        rubberband_available.cache = previous


def _write_sine(path: Path) -> None:
    rate = 44100
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for index in range(rate):
            sample = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * index / rate))
            frames += struct.pack("<h", sample)
        handle.writeframes(frames)


@pytest.mark.skipif(not _libmpv(), reason="libmpv is not installed (mpv-libs / libmpv2)")
def test_live_pitch_keeps_speed(tmp_path):
    previous = rubberband_available.cache
    rubberband_available.cache = None
    try:
        if not rubberband_available():
            pytest.skip("rubberband filter is not available")
        wav = tmp_path / "tone.wav"
        _write_sine(wav)
        player = MpvPlayer((str(wav),), force_window=False, ao="null", vo="null")
        try:
            player.set_pitch(2)
            assert player.pitch_semitones == 2
            assert player.vinyl_mode is False
            assert float(player._player.speed) == pytest.approx(1.0)
        finally:
            player.close()
    finally:
        rubberband_available.cache = previous
