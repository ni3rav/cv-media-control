"""playerctl and wpctl contracts. Subprocess is mocked; nothing is launched."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from gesture_media.players.audio_wpctl import WpctlAudio, playback_stream_ids
from gesture_media.players.base import Capability
from gesture_media.players.mpris import MprisPlayer
from gesture_media.players.registry import build_backends


def _completed(argv, *, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr=stderr)


def _patch_run(monkeypatch, module: str, responder):
    def fake_run(argv, **kwargs):
        return responder(list(argv), kwargs)

    monkeypatch.setattr(f"{module}.subprocess.run", fake_run)


def test_playerctl_commands_and_timeout(monkeypatch):
    seen: list[tuple[list[str], float]] = []

    def responder(argv, kwargs):
        seen.append((argv, kwargs["timeout"]))
        return _completed(argv, stdout="12.5\n")

    _patch_run(monkeypatch, "gesture_media.players.mpris", responder)
    player = MprisPlayer(None, ("vlc",))
    player.play_pause()
    player.next()
    player.previous()
    player.seek(5)
    player.seek(-5)
    player.seek(2.5)
    assert player.get_position() == 12.5
    assert seen == [
        (["playerctl", "play-pause"], 1.0),
        (["playerctl", "next"], 1.0),
        (["playerctl", "previous"], 1.0),
        (["playerctl", "position", "5+"], 1.0),
        (["playerctl", "position", "5-"], 1.0),
        (["playerctl", "position", "2.5+"], 1.0),
        (["playerctl", "position"], 1.0),
    ]


def test_playerctl_respects_player_filter(monkeypatch):
    seen: list[list[str]] = []

    def responder(argv, kwargs):
        seen.append(argv)
        assert kwargs["timeout"] == 1.0
        return _completed(argv, stdout="spotify\n")

    _patch_run(monkeypatch, "gesture_media.players.mpris", responder)
    player = MprisPlayer("spotify", ("spotify",))
    assert player.is_available() is True
    player.seek(-3)
    assert seen[0] == ["playerctl", "--player", "spotify", "-l"]
    assert seen[1] == ["playerctl", "--player", "spotify", "position", "3-"]


def test_playerctl_timeout_and_nonzero_are_swallowed(monkeypatch):
    calls = {"n": 0}

    def responder(argv, kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])
        return _completed(argv, code=1, stderr="no players")

    _patch_run(monkeypatch, "gesture_media.players.mpris", responder)
    player = MprisPlayer(None, ("vlc",))
    player.play_pause()
    player.next()
    assert player.is_available() is False
    assert player.get_position() is None


def test_missing_playerctl_is_swallowed(monkeypatch):
    def responder(argv, kwargs):
        raise FileNotFoundError(argv[0])

    _patch_run(monkeypatch, "gesture_media.players.mpris", responder)
    player = MprisPlayer(None, ("vlc",))
    player.previous()
    assert player.is_available() is False
    assert player.get_position() is None


def test_open_is_detached(monkeypatch):
    seen: dict[str, object] = {}

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("gesture_media.players.mpris.subprocess.Popen", fake_popen)
    MprisPlayer(None, ("vlc", "--started-from-file")).open()
    assert seen["argv"] == ["vlc", "--started-from-file"]
    assert seen["kwargs"]["start_new_session"] is True


def test_set_pitch_is_unsupported_and_does_not_raise(monkeypatch):
    def responder(argv, kwargs):
        raise AssertionError(argv)

    _patch_run(monkeypatch, "gesture_media.players.mpris", responder)
    player = MprisPlayer(None, ("vlc",))
    player.set_pitch(2)
    player.set_pitch(-1)
    assert Capability.PITCH not in player.capabilities


def test_wpctl_commands_and_timeout(monkeypatch):
    seen: list[tuple[list[str], float]] = []

    def responder(argv, kwargs):
        seen.append((argv, kwargs["timeout"]))
        if argv[1] == "get-volume":
            return _completed(argv, stdout="Volume: 0.45 [MUTED]\n")
        return _completed(argv)

    _patch_run(monkeypatch, "gesture_media.players.audio_wpctl", responder)
    audio = WpctlAudio()
    audio.volume_delta(5)
    audio.volume_delta(-3)
    audio.toggle_mute()
    assert audio.get_state() == (45.0, True)
    assert seen == [
        (["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], 1.0),
        (["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"], 1.0),
        (["wpctl", "status"], 1.0),
        (["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+", "-l", "1.0"], 1.0),
        (["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "3%-", "-l", "1.0"], 1.0),
        (["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], 1.0),
        (["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"], 1.0),
        (["wpctl", "status"], 1.0),
        (["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], 1.0),
    ]


def test_unmute_clears_a_muted_playback_stream(monkeypatch):
    status = """
Audio
 └─ Streams:
       107. Firefox
             92. output_FL       > Speaker:playback_FL	[active]
Video
"""
    seen: list[list[str]] = []

    def responder(argv, kwargs):
        seen.append(argv)
        if argv[1] == "status":
            return _completed(argv, stdout=status)
        if argv[1:] == ["get-volume", "@DEFAULT_AUDIO_SINK@"]:
            return _completed(argv, stdout="Volume: 0.40\n")
        if argv[1:] == ["get-volume", "107"]:
            return _completed(argv, stdout="Volume: 1.00 [MUTED]\n")
        return _completed(argv)

    _patch_run(monkeypatch, "gesture_media.players.audio_wpctl", responder)
    audio = WpctlAudio()
    audio.toggle_mute()
    assert ["wpctl", "set-mute", "107", "0"] in seen
    assert ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"] not in seen
    assert playback_stream_ids(status) == ["107"]


def test_wpctl_plain_volume_and_failures(monkeypatch):
    mode = {"kind": "plain"}

    def responder(argv, kwargs):
        if mode["kind"] == "plain":
            return _completed(argv, stdout="Volume: 0.45\n")
        if mode["kind"] == "timeout":
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
        return _completed(argv, code=1, stderr="no sink")

    _patch_run(monkeypatch, "gesture_media.players.audio_wpctl", responder)
    audio = WpctlAudio()
    assert audio.get_state() == (45.0, False)
    mode["kind"] = "timeout"
    audio.volume_delta(1)
    mode["kind"] = "nonzero"
    assert audio.get_state() == (0.0, False)


def _player_config(backend: str, routing: dict | None = None):
    return SimpleNamespace(
        player=SimpleNamespace(
            backend=backend,
            mpris_player=None,
            open_command=("vlc",),
            media=(),
            routing=routing or {},
        ),
        no_preview=True,
    )


class _Dummy:
    def __init__(self, name: str, *, audio: bool) -> None:
        self.name = name
        self.capabilities = frozenset(Capability)
        self.calls: list[object] = []
        self.audio = audio
        self.vinyl_mode = False

    def is_available(self) -> bool:
        return True

    def play_pause(self) -> None:
        self.calls.append("play_pause")

    def next(self) -> None:
        self.calls.append("next")

    def previous(self) -> None:
        self.calls.append("previous")

    def seek(self, seconds: float) -> None:
        self.calls.append(("seek", seconds))

    def set_pitch(self, semitones: float) -> None:
        self.calls.append(("pitch", semitones))

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")

    def get_position(self) -> float | None:
        return 3.0

    def volume_delta(self, percent: float) -> None:
        self.calls.append(("volume", percent))

    def toggle_mute(self) -> None:
        self.calls.append("mute")

    def get_state(self) -> tuple[float, bool]:
        return (40.0, False)


def test_auto_uses_mpris_when_a_player_is_listed(monkeypatch):
    def fake_run(argv, **kwargs):
        assert argv == ["playerctl", "-l"]
        assert kwargs["timeout"] == 1.0
        return _completed(argv, stdout="firefox.instance\n")

    monkeypatch.setattr("gesture_media.players.registry.subprocess.run", fake_run)
    monkeypatch.setattr("gesture_media.players.registry._volume_target", lambda _c, _n: "system")
    from gesture_media.players import registry

    mpris = _Dummy("mpris", audio=False)
    constructed = {"mpv": False}

    def reject_mpv(_config):
        constructed["mpv"] = True
        raise RuntimeError("mpv should not be constructed")

    monkeypatch.setitem(registry.FACTORIES, "mpris", lambda _config: mpris)
    monkeypatch.setitem(registry.FACTORIES, "mpv", reject_mpv)
    player, audio, name = build_backends(_player_config("auto"))
    assert name == "mpris"
    assert isinstance(audio, WpctlAudio)
    assert player.routed_player(Capability.NEXT) is mpris
    assert constructed["mpv"] is False


def test_routing_sends_pitch_to_mpv_and_player_volume(monkeypatch):
    from gesture_media.players import registry

    mpris = _Dummy("mpris", audio=False)
    mpv = _Dummy("mpv", audio=True)
    monkeypatch.setitem(registry.FACTORIES, "mpris", lambda _config: mpris)
    monkeypatch.setitem(registry.FACTORIES, "mpv", lambda _config: mpv)

    def volume_target(_config, name: str) -> str:
        return "player" if name == "mpv" else "system"

    monkeypatch.setattr(registry, "_volume_target", volume_target)
    routing = {"pitch": "mpv", "next": "mpris"}
    player, audio, name = build_backends(_player_config("mpv", routing))
    assert name == "mpv"
    assert audio is mpv
    player.set_pitch(2)
    player.next()
    assert ("pitch", 2) in mpv.calls
    assert "next" in mpris.calls


def test_failed_mpv_does_not_crash_the_registry(monkeypatch):
    from gesture_media.players import registry
    from gesture_media.players.mpv import MpvUnavailable

    monkeypatch.setattr(registry, "_players_listed", lambda: False)
    monkeypatch.setattr(registry, "_volume_target", lambda _c, _n: "system")

    def broken(_config):
        raise MpvUnavailable("libmpv missing")

    monkeypatch.setitem(registry.FACTORIES, "mpv", broken)
    player, audio, name = build_backends(_player_config("auto"))
    assert name == "mpv"
    assert player.is_available() is False
    assert isinstance(audio, WpctlAudio)
    player.play_pause()


def test_best_mpris_player_prefers_one_that_is_playing(monkeypatch):
    from gesture_media.players.registry import best_mpris_player

    def fake_run(argv, **kwargs):
        assert kwargs["timeout"] == 1.0
        if argv == ["playerctl", "-l"]:
            return _completed(argv, stdout="brave.instance\nspotify\n")
        if argv == ["playerctl", "-p", "brave.instance", "status"]:
            return _completed(argv, stdout="Stopped\n")
        if argv == ["playerctl", "-p", "spotify", "status"]:
            return _completed(argv, stdout="Playing\n")
        raise AssertionError(argv)

    monkeypatch.setattr("gesture_media.players.registry.subprocess.run", fake_run)
    assert best_mpris_player() == "spotify"


def test_tray_tooltip_is_latin1():
    from gesture_media.tray import tooltip

    text = tooltip("active", "mpris")
    text.encode("latin-1")
    assert "—" not in text
