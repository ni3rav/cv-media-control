"""Defaults, layering, validation, profiles, and notification policy."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from gesture_media.config import (
    PROFILE_VERSION,
    AppConfig,
    AudioConfig,
    CliOverrides,
    ConfigError,
    ConfigWarning,
    Zone,
    default_config,
    load_config,
    load_profile,
    profile_path,
    resolve_volume_target,
    runtime_dir,
    save_profile,
    should_notify,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_match_the_example_file() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        loaded = load_config(ROOT / "config.yaml", use_profile=False)
    assert loaded == default_config()
    cfg = default_config()
    assert cfg.camera.index == 0
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.camera.fps == 30
    assert cfg.vision.classifier == "mediapipe"
    assert cfg.vision.min_score == pytest.approx(0.6)
    assert cfg.vision.hand == "any"
    assert cfg.vision.zone == Zone(0.15, 0.85, 0.10, 0.90)
    assert cfg.engine.stable_ms == 150
    assert cfg.engine.grace_ms == 200
    assert cfg.engine.tick_ms == 100
    assert cfg.engine.deadzone == pytest.approx(0.05)
    assert cfg.engine.max_disp == pytest.approx(0.25)
    assert cfg.engine.smoothing_alpha == pytest.approx(0.4)
    assert cfg.gestures["Closed_Fist"].action == "play_pause"
    assert cfg.gestures["Closed_Fist"].hold_ms == 350
    assert cfg.gestures["Open_Palm"].action == "mute"
    assert cfg.gestures["Open_Palm"].cooldown_ms == 700
    assert cfg.gestures["Open_Palm"].swipe is None
    assert cfg.gestures["Thumb_Up"].direction == pytest.approx(1)
    assert cfg.gestures["Thumb_Up"].max_rate == pytest.approx(20)
    assert cfg.gestures["Thumb_Down"].direction == pytest.approx(-1)
    assert cfg.gestures["Thumb_Right"].action == "next"
    assert cfg.gestures["Thumb_Right"].hold_ms == 280
    assert cfg.gestures["Thumb_Left"].action == "previous"
    assert cfg.gestures["Pointing_Up"].action == "seek"
    assert cfg.gestures["Pointing_Up"].max_rate == pytest.approx(20)
    assert cfg.gestures["Pointing_Up"].alt_action is None
    assert cfg.gestures["Pointing_Up"].still_action is None
    assert "Victory" not in cfg.gestures
    assert "ILoveYou" not in cfg.gestures
    assert cfg.player.backend == "auto"
    assert cfg.player.mpris_player is None
    assert cfg.player.open_command == ("vlc",)
    assert cfg.player.media == ()
    assert cfg.player.routing == {}
    assert cfg.audio.volume_target == "system"
    assert cfg.audio.volume_target_explicit is False
    assert cfg.pitch.min_semitones == -12
    assert cfg.pitch.max_semitones == 12
    assert cfg.pitch.step == pytest.approx(0.5)
    assert cfg.hud.enabled is True
    assert cfg.hud.show_landmarks is False
    assert cfg.tray.enabled is True
    assert cfg.tray.release_camera_when_paused is False
    assert cfg.pause.gesture == "two_fists"
    assert cfg.pause.hold_ms == 1400
    assert cfg.pause.paused_fps == 5
    assert cfg.notifications.verbosity == "all"
    assert cfg.notifications.update_hz == 5
    assert cfg.notifications.expire_discrete_ms == 3000
    assert cfg.notifications.expire_continuous_ms == 1500
    assert cfg.notifications.live_updates is True
    assert cfg.notifications.final_value is True
    assert cfg.notifications.on_start is True
    assert cfg.notifications.on_toggle is True
    assert cfg.notifications.on_error is True
    assert cfg.calibration_active is False
    assert cfg.source is None
    assert cfg.no_preview is False
    assert cfg.verbose is False
    assert PROFILE_VERSION == 1
    assert issubclass(ConfigError, ValueError)


def test_yaml_overrides_and_partial_gesture(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
gestures:
  Open_Palm: {hold_ms: 800}
vision:
  min_score: 0.7
  hand: left
player:
  backend: mpris
  media: [a.mp3, b.mp3]
  routing: {pitch: mpv}
audio:
  volume_target: player
""",
    )
    cfg = load_config(path, use_profile=False)
    assert cfg.gestures["Open_Palm"].hold_ms == 800
    assert cfg.gestures["Open_Palm"].action == "mute"
    assert cfg.gestures["Open_Palm"].cooldown_ms == 700
    assert cfg.gestures["Open_Palm"].swipe is None
    assert cfg.gestures["Thumb_Up"].max_rate == pytest.approx(20)
    assert cfg.vision.min_score == pytest.approx(0.7)
    assert cfg.vision.hand == "left"
    assert cfg.player.backend == "mpris"
    assert cfg.player.media == ("a.mp3", "b.mp3")
    assert cfg.player.routing == {"pitch": "mpv"}
    assert cfg.audio.volume_target == "player"
    assert cfg.audio.volume_target_explicit is True


def test_unknown_keys_warn_and_do_not_fail(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
camera:
  index: 2
  mystery: true
future: 1
""",
    )
    with pytest.warns(ConfigWarning):
        cfg = load_config(path, use_profile=False)
    assert cfg.camera.index == 2
    assert cfg.engine.stable_ms == 150


@pytest.mark.parametrize(
    "text",
    [
        "vision: {min_score: 1.5}",
        "vision: {min_score: -0.1}",
        "vision: {hand: both}",
        "vision: {classifier: opencv}",
        "vision: {zone: {x_min: 0.9, x_max: 0.2, y_min: 0.1, y_max: 0.9}}",
        "vision: {zone: {x_min: -0.1, x_max: 0.5, y_min: 0.1, y_max: 0.9}}",
        "engine: {stable_ms: 0}",
        "engine: {stable_ms: true}",
        "engine: {deadzone: 0.3}",
        "engine: {deadzone: -0.01}",
        "engine: {smoothing_alpha: 0}",
        "engine: {smoothing_alpha: 1.2}",
        "player: {backend: vlc}",
        "audio: {volume_target: both}",
        "notifications: {verbosity: loud}",
        "pitch: {min_semitones: 3, max_semitones: 3}",
        "pitch: {step: 0}",
        "camera: {fps: 0}",
        "camera: {width: -1}",
        "pause: {hold_ms: 0}",
        "gestures: {Open_Palm: {hold_ms: 0}}",
        "gestures: {Victory: {max_rate: 0}}",
        "[]",
    ],
)
def test_invalid_values_raise(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ConfigError):
        load_config(path, use_profile=False)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml", use_profile=False)


def test_profile_overrides_yaml_and_cli_overrides_profile(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path,
        """
camera: {index: 1}
engine: {deadzone: 0.08, max_disp: 0.30}
vision:
  min_score: 0.6
  zone: {x_min: 0.2, x_max: 0.8, y_min: 0.2, y_max: 0.8}
player: {backend: mpris}
""",
    )
    profile = _write(
        tmp_path,
        """
version: 1
deadzone: 0.04
max_disp: 0.22
min_score: 0.75
zone: {x_min: 0.12, x_max: 0.9, y_min: 0.11, y_max: 0.88}
""",
        name="profile.yaml",
    )
    cli = CliOverrides(
        camera_index=4,
        source="clip.mp4",
        player_backend="mpv",
        media=("a.mp3",),
        no_preview=True,
        verbose=True,
    )
    cfg = load_config(config_path, profile_file=profile, cli=cli)
    assert cfg.engine.deadzone == pytest.approx(0.04)
    assert cfg.engine.max_disp == pytest.approx(0.22)
    assert cfg.vision.min_score == pytest.approx(0.75)
    assert cfg.vision.zone == Zone(0.12, 0.9, 0.11, 0.88)
    assert cfg.calibration_active is True
    assert cfg.camera.index == 4
    assert cfg.player.backend == "mpv"
    assert cfg.player.media == ("a.mp3",)
    assert cfg.source == "clip.mp4"
    assert cfg.no_preview is True
    assert cfg.verbose is True

    skipped = load_config(
        config_path,
        profile_file=profile,
        cli=replace(cli, no_calibration=True),
    )
    assert skipped.calibration_active is False
    assert skipped.engine.deadzone == pytest.approx(0.08)
    assert skipped.vision.min_score == pytest.approx(0.6)
    assert skipped.camera.index == 4
    assert skipped.player.backend == "mpv"


def test_profiles_missing_corrupt_old_and_roundtrip(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert load_profile(missing) is None
    assert caught == []

    corrupt = _write(tmp_path, ":\n  - [", name="bad.yaml")
    with pytest.warns(ConfigWarning):
        assert load_profile(corrupt) is None

    old = _write(
        tmp_path,
        """
version: 0
deadzone: 0.04
max_disp: 0.2
zone: {x_min: 0.1, x_max: 0.9, y_min: 0.1, y_max: 0.9}
""",
        name="old.yaml",
    )
    with pytest.warns(ConfigWarning, match="unsupported version"):
        assert load_profile(old) is None

    invalid = _write(
        tmp_path,
        """
version: 1
deadzone: 0.4
max_disp: 0.2
zone: {x_min: 0.1, x_max: 0.9, y_min: 0.1, y_max: 0.9}
""",
        name="invalid.yaml",
    )
    with pytest.warns(ConfigWarning):
        assert load_profile(invalid) is None

    extra = _write(
        tmp_path,
        """
version: 1
deadzone: 0.04
max_disp: 0.2
zone: {x_min: 0.1, x_max: 0.9, y_min: 0.1, y_max: 0.9}
extra: 1
""",
        name="extra.yaml",
    )
    with pytest.warns(ConfigWarning, match="unknown calibration profile key"):
        loaded = load_profile(extra)
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.min_score is None

    assert loaded is not None
    destination = tmp_path / "nested" / "saved.yaml"
    save_profile(destination, replace(loaded, min_score=0.55))
    again = load_profile(destination)
    assert again is not None
    assert again == replace(loaded, min_score=0.55)


def test_profile_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert profile_path() == tmp_path / "gesture-media" / "profile.yaml"
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert runtime_dir() == Path("/run/user/1000")
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert runtime_dir() == Path.home() / ".cache" / "gesture-media"
    cfg = load_config(use_profile=True)
    assert cfg.calibration_active is False


def test_resolve_volume_target(tmp_path: Path) -> None:
    cfg = default_config()
    assert resolve_volume_target(cfg, "mpv") == "player"
    assert resolve_volume_target(cfg, "mpris") == "system"
    assert resolve_volume_target(cfg, "auto") == "system"
    explicit_system = replace(cfg, audio=AudioConfig("system", True))
    explicit_player = replace(cfg, audio=AudioConfig("player", True))
    assert resolve_volume_target(explicit_system, "mpv") == "system"
    assert resolve_volume_target(explicit_player, "mpris") == "player"

    path = _write(tmp_path, "audio: {volume_target: player}\n")
    loaded = load_config(path, use_profile=False)
    assert loaded.audio.volume_target_explicit is True
    assert resolve_volume_target(loaded, "mpris") == "player"
    omitted = _write(tmp_path, "audio: {}\n", name="empty-audio.yaml")
    implicit = load_config(omitted, use_profile=False)
    assert implicit.audio.volume_target_explicit is False
    assert resolve_volume_target(implicit, "mpv") == "player"


def test_should_notify_matrix() -> None:
    cfg = default_config()
    notes = cfg.notifications
    assert should_notify(cfg, "app_on") is True
    assert should_notify(cfg, "play_pause") is True
    assert should_notify(cfg, "calibration") is True
    assert should_notify(cfg, "volume") is True
    assert should_notify(cfg, "not-an-event") is False

    disabled = replace(cfg, notifications=replace(notes, enabled=False))
    assert should_notify(disabled, "app_on") is False
    silent = replace(cfg, notifications=replace(notes, verbosity="off"))
    assert should_notify(silent, "error") is False

    important = replace(cfg, notifications=replace(notes, verbosity="important"))
    for event in ("app_on", "app_off", "pause", "resume", "error"):
        assert should_notify(important, event) is True
    assert should_notify(important, "volume") is False
    assert should_notify(important, "play_pause") is False

    no_start = replace(cfg, notifications=replace(notes, on_start=False))
    assert should_notify(no_start, "app_on") is False
    assert should_notify(no_start, "app_off") is True
    important_no_start = replace(
        cfg,
        notifications=replace(notes, verbosity="important", on_start=False),
    )
    assert should_notify(important_no_start, "app_on") is False

    no_toggle = replace(cfg, notifications=replace(notes, on_toggle=False))
    assert should_notify(no_toggle, "pause") is False
    assert should_notify(no_toggle, "resume") is False
    no_error = replace(cfg, notifications=replace(notes, on_error=False))
    assert should_notify(no_error, "error") is False

    quiet_volume = replace(
        cfg,
        notifications=replace(notes, per_action={"volume": False, "seek": True}),
    )
    assert should_notify(quiet_volume, "volume") is False
    assert should_notify(quiet_volume, "seek") is True
    assert should_notify(quiet_volume, "pitch") is True

    forced = replace(
        cfg,
        notifications=replace(notes, verbosity="important", per_action={"volume": True}),
    )
    assert should_notify(forced, "volume") is False
    gated = replace(
        cfg,
        notifications=replace(notes, on_toggle=False, per_action={"pause": True}),
    )
    assert should_notify(gated, "pause") is False


def test_cli_backend_must_be_known() -> None:
    with pytest.raises(ConfigError):
        load_config(use_profile=False, cli=CliOverrides(player_backend="vlc"))
    assert isinstance(default_config(), AppConfig)
