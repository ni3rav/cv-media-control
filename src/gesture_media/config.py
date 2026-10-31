"""YAML configuration, calibration profiles, and notification policy.

Precedence, lowest to highest: built-in defaults, YAML file, calibration
profile, CLI overrides. Unknown keys warn. Invalid values raise ConfigError.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

PROFILE_VERSION = 1

_DISCRETE_ACTIONS = frozenset({"play_pause", "mute", "next", "previous"})
_CONTINUOUS_ACTIONS = frozenset({"volume", "seek", "pitch"})
_STILL_ACTIONS = frozenset({"open_player", "pitch_reset"})
_HANDS = frozenset({"any", "left", "right"})
_CLASSIFIERS = frozenset({"mediapipe", "rules"})
_BACKENDS = frozenset({"auto", "mpv", "mpris"})
_VOLUME_TARGETS = frozenset({"system", "player"})
_VERBOSITY = frozenset({"all", "important", "off"})
_IMPORTANT_EVENTS = frozenset({"app_on", "app_off", "pause", "resume", "error"})
_NOTIFY_EVENTS = _IMPORTANT_EVENTS | frozenset(
    {
        "play_pause",
        "mute",
        "next",
        "previous",
        "volume",
        "seek",
        "pitch",
        "pitch_reset",
        "open_player",
        "unsupported",
        "calibration",
    }
)
_ROOT_KEYS = frozenset(
    {
        "camera",
        "vision",
        "engine",
        "gestures",
        "player",
        "audio",
        "pitch",
        "hud",
        "tray",
        "pause",
        "notifications",
    }
)


class ConfigError(ValueError):
    """The configuration file or an override is invalid."""


class ConfigWarning(UserWarning):
    """A configuration file or profile has an unknown or unusable entry."""


@dataclass(frozen=True, slots=True)
class Zone:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True, slots=True)
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int


@dataclass(frozen=True, slots=True)
class VisionConfig:
    classifier: str
    min_score: float
    hand: str
    zone: Zone


@dataclass(frozen=True, slots=True)
class EngineConfig:
    stable_ms: int
    grace_ms: int
    tick_ms: int
    deadzone: float
    max_disp: float
    smoothing_alpha: float


@dataclass(frozen=True, slots=True)
class GestureSpec:
    action: str
    hold_ms: int | None = None
    cooldown_ms: int | None = None
    max_rate: float | None = None
    still_action: str | None = None
    still_ms: int | None = None
    direction: float | None = None
    alt_action: str | None = None
    alt_max_rate: float | None = None
    swipe: str | None = None


@dataclass(frozen=True, slots=True)
class PlayerConfig:
    backend: str
    mpris_player: str | None
    open_command: tuple[str, ...]
    media: tuple[str, ...]
    routing: dict[str, str]


@dataclass(frozen=True, slots=True)
class AudioConfig:
    volume_target: str
    volume_target_explicit: bool


@dataclass(frozen=True, slots=True)
class PitchConfig:
    min_semitones: float
    max_semitones: float
    step: float


@dataclass(frozen=True, slots=True)
class HudConfig:
    enabled: bool
    show_landmarks: bool


@dataclass(frozen=True, slots=True)
class TrayConfig:
    enabled: bool
    release_camera_when_paused: bool


@dataclass(frozen=True, slots=True)
class PauseConfig:
    gesture: str
    hold_ms: int
    paused_fps: int


@dataclass(frozen=True, slots=True)
class NotificationsConfig:
    enabled: bool
    verbosity: str
    update_hz: int
    expire_discrete_ms: int
    expire_continuous_ms: int
    live_updates: bool
    final_value: bool
    on_start: bool
    on_toggle: bool
    on_error: bool
    per_action: dict[str, bool]


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    version: int
    deadzone: float
    max_disp: float
    zone: Zone
    min_score: float | None = None


@dataclass(frozen=True, slots=True)
class CliOverrides:
    camera_index: int | None = None
    source: str | None = None
    player_backend: str | None = None
    media: tuple[str, ...] | None = None
    no_preview: bool = False
    no_calibration: bool = False
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraConfig
    vision: VisionConfig
    engine: EngineConfig
    gestures: dict[str, GestureSpec]
    player: PlayerConfig
    audio: AudioConfig
    pitch: PitchConfig
    hud: HudConfig
    tray: TrayConfig
    pause: PauseConfig
    notifications: NotificationsConfig
    calibration_active: bool = False
    source: str | None = None
    no_preview: bool = False
    verbose: bool = False


def default_config() -> AppConfig:
    """Built-in defaults. The same object shape as a loaded config."""
    config = AppConfig(
        camera=CameraConfig(index=0, width=640, height=480, fps=30),
        vision=VisionConfig(
            classifier="mediapipe",
            min_score=0.6,
            hand="any",
            zone=Zone(0.15, 0.85, 0.10, 0.90),
        ),
        engine=EngineConfig(
            stable_ms=150,
            grace_ms=200,
            tick_ms=100,
            deadzone=0.05,
            max_disp=0.25,
            smoothing_alpha=0.4,
        ),
        gestures={
            "Closed_Fist": GestureSpec("play_pause", hold_ms=350, cooldown_ms=700),
            "Open_Palm": GestureSpec("mute", hold_ms=350, cooldown_ms=700),
            "Thumb_Up": GestureSpec("volume", max_rate=20.0, direction=1.0),
            "Thumb_Down": GestureSpec("volume", max_rate=20.0, direction=-1.0),
            "Thumb_Right": GestureSpec("next", hold_ms=280, cooldown_ms=700),
            "Thumb_Left": GestureSpec("previous", hold_ms=280, cooldown_ms=700),
            "Pointing_Up": GestureSpec("seek", max_rate=20.0),
        },
        player=PlayerConfig(
            backend="auto",
            mpris_player=None,
            open_command=("vlc",),
            media=(),
            routing={},
        ),
        audio=AudioConfig(volume_target="system", volume_target_explicit=False),
        pitch=PitchConfig(min_semitones=-12.0, max_semitones=12.0, step=0.5),
        hud=HudConfig(enabled=True, show_landmarks=False),
        tray=TrayConfig(enabled=True, release_camera_when_paused=False),
        pause=PauseConfig(gesture="two_fists", hold_ms=1400, paused_fps=5),
        notifications=NotificationsConfig(
            enabled=True,
            verbosity="all",
            update_hz=5,
            expire_discrete_ms=3000,
            expire_continuous_ms=1500,
            live_updates=True,
            final_value=True,
            on_start=True,
            on_toggle=True,
            on_error=True,
            per_action={"volume": True, "seek": True},
        ),
    )
    _validate(config)
    return config


def load_config(
    path: Path | None = None,
    *,
    profile_file: Path | None = None,
    use_profile: bool = True,
    cli: CliOverrides | None = None,
) -> AppConfig:
    """Load defaults, then YAML, profile, and CLI, in that order.

    ``path is None`` skips the YAML file. A missing explicit path is an error.
    A missing profile is skipped. ``cli.no_calibration`` skips it too.
    """
    config = default_config()
    if path is not None:
        config = _apply_yaml(config, Path(path))
    skip_profile = not use_profile or (cli is not None and cli.no_calibration)
    if not skip_profile:
        profile = load_profile(Path(profile_file) if profile_file is not None else profile_path())
        if profile is not None:
            config = _apply_profile(config, profile)
    if cli is not None:
        config = _apply_cli(config, cli)
    _validate(config)
    return config


def load_profile(path: Path) -> CalibrationProfile | None:
    """Load a version-1 profile. A missing file returns None. Corrupt or old data warns."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        warnings.warn(
            f"cannot read calibration profile {path}: {exc}",
            ConfigWarning,
            stacklevel=2,
        )
        return None
    try:
        return _parse_profile(text)
    except (ConfigError, yaml.YAMLError, ValueError) as exc:
        warnings.warn(
            f"ignoring calibration profile {path}: {exc}",
            ConfigWarning,
            stacklevel=2,
        )
        return None


def save_profile(path: Path, profile: CalibrationProfile) -> None:
    """Write a calibration profile, creating parent directories."""
    path = Path(path)
    payload: dict[str, Any] = {
        "version": profile.version,
        "deadzone": profile.deadzone,
        "max_disp": profile.max_disp,
        "zone": {
            "x_min": profile.zone.x_min,
            "x_max": profile.zone.x_max,
            "y_min": profile.zone.y_min,
            "y_max": profile.zone.y_max,
        },
    }
    if profile.min_score is not None:
        payload["min_score"] = profile.min_score
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def profile_path() -> Path:
    """``$XDG_CONFIG_HOME/gesture-media/profile.yaml`` or ``~/.config/...``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "gesture-media" / "profile.yaml"


def runtime_dir() -> Path:
    """``$XDG_RUNTIME_DIR`` or ``~/.cache/gesture-media``."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg)
    return Path.home() / ".cache" / "gesture-media"


def resolve_volume_target(config: AppConfig, backend_name: str) -> str:
    """Use the explicit target, otherwise player volume only for mpv."""
    if config.audio.volume_target_explicit:
        return config.audio.volume_target
    if backend_name == "mpv":
        return "player"
    return "system"


def should_notify(config: AppConfig, event: str) -> bool:
    """Return whether ``event`` should produce a desktop notification."""
    notes = config.notifications
    if not notes.enabled or notes.verbosity == "off":
        return False
    if notes.verbosity == "important":
        if event not in _IMPORTANT_EVENTS:
            return False
    elif notes.verbosity == "all":
        if event not in _NOTIFY_EVENTS:
            return False
        if event in notes.per_action and not notes.per_action[event]:
            return False
    else:
        return False
    if event == "app_on" and not notes.on_start:
        return False
    if event in {"pause", "resume"} and not notes.on_toggle:
        return False
    if event == "error" and not notes.on_error:
        return False
    return True


def _apply_yaml(config: AppConfig, path: Path) -> AppConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return config
    data = _mapping(loaded, "config")
    _warn_unknown("", data, _ROOT_KEYS)
    audio = config.audio
    if "audio" in data:
        audio = _apply_audio(audio, data["audio"])
    return replace(
        config,
        camera=_apply_camera(config.camera, data["camera"]) if "camera" in data else config.camera,
        vision=_apply_vision(config.vision, data["vision"]) if "vision" in data else config.vision,
        engine=_apply_engine(config.engine, data["engine"]) if "engine" in data else config.engine,
        gestures=_apply_gestures(config.gestures, data["gestures"])
        if "gestures" in data
        else config.gestures,
        player=_apply_player(config.player, data["player"]) if "player" in data else config.player,
        audio=audio,
        pitch=_apply_pitch(config.pitch, data["pitch"]) if "pitch" in data else config.pitch,
        hud=_apply_hud(config.hud, data["hud"]) if "hud" in data else config.hud,
        tray=_apply_tray(config.tray, data["tray"]) if "tray" in data else config.tray,
        pause=_apply_pause(config.pause, data["pause"]) if "pause" in data else config.pause,
        notifications=_apply_notifications(config.notifications, data["notifications"])
        if "notifications" in data
        else config.notifications,
    )


def _apply_profile(config: AppConfig, profile: CalibrationProfile) -> AppConfig:
    vision = replace(config.vision, zone=profile.zone)
    if profile.min_score is not None:
        vision = replace(vision, min_score=profile.min_score)
    engine = replace(config.engine, deadzone=profile.deadzone, max_disp=profile.max_disp)
    return replace(config, vision=vision, engine=engine, calibration_active=True)


def _apply_cli(config: AppConfig, cli: CliOverrides) -> AppConfig:
    camera = config.camera
    player = config.player
    if cli.camera_index is not None:
        camera = replace(camera, index=_as_int(cli.camera_index, "cli.camera_index"))
    if cli.player_backend is not None:
        if not isinstance(cli.player_backend, str):
            raise ConfigError("cli.player_backend must be a string")
        player = replace(player, backend=cli.player_backend)
    if cli.media is not None:
        player = replace(player, media=_string_tuple(cli.media, "cli.media"))
    source = config.source
    if cli.source is not None:
        source = _as_str(cli.source, "cli.source")
    if not isinstance(cli.no_preview, bool) or not isinstance(cli.verbose, bool):
        raise ConfigError("cli.no_preview and cli.verbose must be booleans")
    return replace(
        config,
        camera=camera,
        player=player,
        source=source,
        no_preview=cli.no_preview,
        verbose=cli.verbose,
    )


def _apply_camera(current: CameraConfig, raw: object) -> CameraConfig:
    data = _mapping(raw, "camera")
    _warn_unknown("camera", data, {"index", "width", "height", "fps"})
    return CameraConfig(
        index=_pick_int(data, "index", current.index, "camera.index"),
        width=_pick_int(data, "width", current.width, "camera.width"),
        height=_pick_int(data, "height", current.height, "camera.height"),
        fps=_pick_int(data, "fps", current.fps, "camera.fps"),
    )


def _apply_vision(current: VisionConfig, raw: object) -> VisionConfig:
    data = _mapping(raw, "vision")
    _warn_unknown("vision", data, {"classifier", "min_score", "hand", "zone"})
    if "zone" in data:
        zone = _parse_zone(data["zone"], "vision.zone", current.zone)
    else:
        zone = current.zone
    return VisionConfig(
        classifier=_pick_str(data, "classifier", current.classifier, "vision.classifier"),
        min_score=_pick_float(data, "min_score", current.min_score, "vision.min_score"),
        hand=_pick_str(data, "hand", current.hand, "vision.hand"),
        zone=zone,
    )


def _apply_engine(current: EngineConfig, raw: object) -> EngineConfig:
    data = _mapping(raw, "engine")
    _warn_unknown(
        "engine",
        data,
        {"stable_ms", "grace_ms", "tick_ms", "deadzone", "max_disp", "smoothing_alpha"},
    )
    return EngineConfig(
        stable_ms=_pick_int(data, "stable_ms", current.stable_ms, "engine.stable_ms"),
        grace_ms=_pick_int(data, "grace_ms", current.grace_ms, "engine.grace_ms"),
        tick_ms=_pick_int(data, "tick_ms", current.tick_ms, "engine.tick_ms"),
        deadzone=_pick_float(data, "deadzone", current.deadzone, "engine.deadzone"),
        max_disp=_pick_float(data, "max_disp", current.max_disp, "engine.max_disp"),
        smoothing_alpha=_pick_float(
            data,
            "smoothing_alpha",
            current.smoothing_alpha,
            "engine.smoothing_alpha",
        ),
    )


def _apply_gestures(
    current: dict[str, GestureSpec],
    raw: object,
) -> dict[str, GestureSpec]:
    data = _mapping(raw, "gestures")
    merged = dict(current)
    fields = {
        "action",
        "hold_ms",
        "cooldown_ms",
        "max_rate",
        "still_action",
        "still_ms",
        "direction",
        "alt_action",
        "alt_max_rate",
        "swipe",
    }
    for name, spec_raw in data.items():
        if not isinstance(name, str) or not name:
            raise ConfigError("gesture names must be non-empty strings")
        spec_map = _mapping(spec_raw, f"gestures.{name}")
        _warn_unknown(f"gestures.{name}", spec_map, fields)
        base = merged.get(name)
        action = _pick_str(
            spec_map,
            "action",
            None if base is None else base.action,
            f"gestures.{name}.action",
        )
        if action is None:
            raise ConfigError(f"gestures.{name} requires action")
        merged[name] = GestureSpec(
            action=action,
            hold_ms=_pick_optional_int(
                spec_map,
                "hold_ms",
                None if base is None else base.hold_ms,
                f"gestures.{name}.hold_ms",
            ),
            cooldown_ms=_pick_optional_int(
                spec_map,
                "cooldown_ms",
                None if base is None else base.cooldown_ms,
                f"gestures.{name}.cooldown_ms",
            ),
            max_rate=_pick_optional_float(
                spec_map,
                "max_rate",
                None if base is None else base.max_rate,
                f"gestures.{name}.max_rate",
            ),
            still_action=_pick_optional_str(
                spec_map,
                "still_action",
                None if base is None else base.still_action,
                f"gestures.{name}.still_action",
            ),
            still_ms=_pick_optional_int(
                spec_map,
                "still_ms",
                None if base is None else base.still_ms,
                f"gestures.{name}.still_ms",
            ),
            direction=_pick_optional_float(
                spec_map,
                "direction",
                None if base is None else base.direction,
                f"gestures.{name}.direction",
            ),
            alt_action=_pick_optional_str(
                spec_map,
                "alt_action",
                None if base is None else base.alt_action,
                f"gestures.{name}.alt_action",
            ),
            alt_max_rate=_pick_optional_float(
                spec_map,
                "alt_max_rate",
                None if base is None else base.alt_max_rate,
                f"gestures.{name}.alt_max_rate",
            ),
            swipe=_pick_optional_str(
                spec_map,
                "swipe",
                None if base is None else base.swipe,
                f"gestures.{name}.swipe",
            ),
        )
    return merged


def _apply_player(current: PlayerConfig, raw: object) -> PlayerConfig:
    data = _mapping(raw, "player")
    _warn_unknown(
        "player",
        data,
        {"backend", "mpris_player", "open_command", "media", "routing"},
    )
    if "mpris_player" in data and data["mpris_player"] is not None:
        mpris: str | None = _as_str(data["mpris_player"], "player.mpris_player")
    elif "mpris_player" in data:
        mpris = None
    else:
        mpris = current.mpris_player
    return PlayerConfig(
        backend=_pick_str(data, "backend", current.backend, "player.backend"),
        mpris_player=mpris,
        open_command=_string_tuple(data["open_command"], "player.open_command")
        if "open_command" in data
        else current.open_command,
        media=_string_tuple(data["media"], "player.media") if "media" in data else current.media,
        routing=_routing(data["routing"]) if "routing" in data else dict(current.routing),
    )


def _apply_audio(current: AudioConfig, raw: object) -> AudioConfig:
    data = _mapping(raw, "audio")
    _warn_unknown("audio", data, {"volume_target"})
    if "volume_target" not in data:
        return current
    return AudioConfig(
        volume_target=_as_str(data["volume_target"], "audio.volume_target"),
        volume_target_explicit=True,
    )


def _apply_pitch(current: PitchConfig, raw: object) -> PitchConfig:
    data = _mapping(raw, "pitch")
    _warn_unknown("pitch", data, {"min_semitones", "max_semitones", "step"})
    return PitchConfig(
        min_semitones=_pick_float(
            data, "min_semitones", current.min_semitones, "pitch.min_semitones"
        ),
        max_semitones=_pick_float(
            data, "max_semitones", current.max_semitones, "pitch.max_semitones"
        ),
        step=_pick_float(data, "step", current.step, "pitch.step"),
    )


def _apply_hud(current: HudConfig, raw: object) -> HudConfig:
    data = _mapping(raw, "hud")
    _warn_unknown("hud", data, {"enabled", "show_landmarks"})
    return HudConfig(
        enabled=_pick_bool(data, "enabled", current.enabled, "hud.enabled"),
        show_landmarks=_pick_bool(
            data, "show_landmarks", current.show_landmarks, "hud.show_landmarks"
        ),
    )


def _apply_tray(current: TrayConfig, raw: object) -> TrayConfig:
    data = _mapping(raw, "tray")
    _warn_unknown("tray", data, {"enabled", "release_camera_when_paused"})
    return TrayConfig(
        enabled=_pick_bool(data, "enabled", current.enabled, "tray.enabled"),
        release_camera_when_paused=_pick_bool(
            data,
            "release_camera_when_paused",
            current.release_camera_when_paused,
            "tray.release_camera_when_paused",
        ),
    )


def _apply_pause(current: PauseConfig, raw: object) -> PauseConfig:
    data = _mapping(raw, "pause")
    _warn_unknown("pause", data, {"gesture", "hold_ms", "paused_fps"})
    return PauseConfig(
        gesture=_pick_str(data, "gesture", current.gesture, "pause.gesture"),
        hold_ms=_pick_int(data, "hold_ms", current.hold_ms, "pause.hold_ms"),
        paused_fps=_pick_int(data, "paused_fps", current.paused_fps, "pause.paused_fps"),
    )


def _apply_notifications(current: NotificationsConfig, raw: object) -> NotificationsConfig:
    data = _mapping(raw, "notifications")
    _warn_unknown(
        "notifications",
        data,
        {
            "enabled",
            "verbosity",
            "update_hz",
            "expire_discrete_ms",
            "expire_continuous_ms",
            "live_updates",
            "final_value",
            "on_start",
            "on_toggle",
            "on_error",
            "per_action",
        },
    )
    per_action = dict(current.per_action)
    if "per_action" in data:
        per_action = _per_action(data["per_action"])
    return NotificationsConfig(
        enabled=_pick_bool(data, "enabled", current.enabled, "notifications.enabled"),
        verbosity=_pick_str(data, "verbosity", current.verbosity, "notifications.verbosity"),
        update_hz=_pick_int(data, "update_hz", current.update_hz, "notifications.update_hz"),
        expire_discrete_ms=_pick_int(
            data,
            "expire_discrete_ms",
            current.expire_discrete_ms,
            "notifications.expire_discrete_ms",
        ),
        expire_continuous_ms=_pick_int(
            data,
            "expire_continuous_ms",
            current.expire_continuous_ms,
            "notifications.expire_continuous_ms",
        ),
        live_updates=_pick_bool(
            data, "live_updates", current.live_updates, "notifications.live_updates"
        ),
        final_value=_pick_bool(
            data, "final_value", current.final_value, "notifications.final_value"
        ),
        on_start=_pick_bool(data, "on_start", current.on_start, "notifications.on_start"),
        on_toggle=_pick_bool(data, "on_toggle", current.on_toggle, "notifications.on_toggle"),
        on_error=_pick_bool(data, "on_error", current.on_error, "notifications.on_error"),
        per_action=per_action,
    )


def _parse_profile(text: str) -> CalibrationProfile:
    loaded = yaml.safe_load(text)
    data = _mapping(loaded, "calibration profile")
    known = {"version", "deadzone", "max_disp", "zone", "min_score"}
    for key in data:
        if key not in known:
            warnings.warn(f"unknown calibration profile key '{key}'", ConfigWarning, stacklevel=3)
    if "version" not in data:
        raise ConfigError("missing version")
    version = data["version"]
    if type(version) is not int:
        raise ConfigError(f"unsupported version {version!r}; expected {PROFILE_VERSION}")
    if version != PROFILE_VERSION:
        raise ConfigError(f"unsupported version {version}; expected {PROFILE_VERSION}")
    for key in ("deadzone", "max_disp", "zone"):
        if key not in data:
            raise ConfigError(f"missing {key}")
    deadzone = _as_float(data["deadzone"], "deadzone")
    max_disp = _as_float(data["max_disp"], "max_disp")
    zone = _parse_zone(data["zone"], "zone", fallback=None)
    min_score = None
    if "min_score" in data and data["min_score"] is not None:
        min_score = _as_float(data["min_score"], "min_score")
    profile = CalibrationProfile(version, deadzone, max_disp, zone, min_score)
    _validate_profile(profile)
    return profile


def _validate_profile(profile: CalibrationProfile) -> None:
    if profile.deadzone < 0 or profile.deadzone >= profile.max_disp:
        raise ConfigError(
            f"deadzone {profile.deadzone} must satisfy 0 <= deadzone < max_disp "
            f"({profile.max_disp})"
        )
    _validate_zone(profile.zone, "zone")
    if profile.min_score is not None and not 0.0 <= profile.min_score <= 1.0:
        raise ConfigError(f"min_score {profile.min_score} must be between 0 and 1")


def _validate(config: AppConfig) -> None:
    camera = config.camera
    if camera.index < 0:
        raise ConfigError(f"camera.index {camera.index} must be >= 0")
    _require_positive(camera.width, "camera.width")
    _require_positive(camera.height, "camera.height")
    _require_positive(camera.fps, "camera.fps")

    vision = config.vision
    if vision.classifier not in _CLASSIFIERS:
        raise ConfigError(f"vision.classifier {vision.classifier!r} must be mediapipe or rules")
    if not 0.0 <= vision.min_score <= 1.0:
        raise ConfigError(f"vision.min_score {vision.min_score} must be between 0 and 1")
    if vision.hand not in _HANDS:
        raise ConfigError(f"vision.hand {vision.hand!r} must be any, left, or right")
    _validate_zone(vision.zone, "vision.zone")

    engine = config.engine
    _require_positive(engine.stable_ms, "engine.stable_ms")
    _require_positive(engine.grace_ms, "engine.grace_ms")
    _require_positive(engine.tick_ms, "engine.tick_ms")
    if engine.deadzone < 0 or engine.deadzone >= engine.max_disp:
        raise ConfigError(
            f"engine.deadzone {engine.deadzone} must satisfy 0 <= deadzone < max_disp "
            f"({engine.max_disp})"
        )
    if not 0.0 < engine.smoothing_alpha <= 1.0:
        raise ConfigError(f"engine.smoothing_alpha {engine.smoothing_alpha} must be in (0, 1]")

    for name, spec in config.gestures.items():
        _validate_gesture(name, spec)

    if config.player.backend not in _BACKENDS:
        raise ConfigError(f"player.backend {config.player.backend!r} must be auto, mpv, or mpris")
    if config.audio.volume_target not in _VOLUME_TARGETS:
        raise ConfigError(
            f"audio.volume_target {config.audio.volume_target!r} must be system or player"
        )

    pitch = config.pitch
    if pitch.min_semitones >= pitch.max_semitones:
        raise ConfigError(
            f"pitch.min_semitones {pitch.min_semitones} must be < max_semitones "
            f"{pitch.max_semitones}"
        )
    if pitch.step <= 0:
        raise ConfigError(f"pitch.step {pitch.step} must be positive")

    _require_positive(config.pause.hold_ms, "pause.hold_ms")
    _require_positive(config.pause.paused_fps, "pause.paused_fps")
    if not config.pause.gesture:
        raise ConfigError("pause.gesture must be a non-empty string")

    notes = config.notifications
    if notes.verbosity not in _VERBOSITY:
        raise ConfigError(
            f"notifications.verbosity {notes.verbosity!r} must be all, important, or off"
        )
    _require_positive(notes.update_hz, "notifications.update_hz")
    _require_positive(notes.expire_discrete_ms, "notifications.expire_discrete_ms")
    _require_positive(notes.expire_continuous_ms, "notifications.expire_continuous_ms")


def _validate_gesture(name: str, spec: GestureSpec) -> None:
    label = f"gestures.{name}"
    if spec.action in _DISCRETE_ACTIONS:
        if spec.hold_ms is None or spec.hold_ms <= 0:
            raise ConfigError(f"{label}.hold_ms must be positive")
        if spec.cooldown_ms is None or spec.cooldown_ms <= 0:
            raise ConfigError(f"{label}.cooldown_ms must be positive")
    elif spec.action in _CONTINUOUS_ACTIONS:
        if spec.max_rate is None or spec.max_rate <= 0:
            raise ConfigError(f"{label}.max_rate must be positive")
    else:
        raise ConfigError(f"{label}.action {spec.action!r} is not a known action")
    if spec.hold_ms is not None and spec.hold_ms <= 0:
        raise ConfigError(f"{label}.hold_ms must be positive")
    if spec.cooldown_ms is not None and spec.cooldown_ms <= 0:
        raise ConfigError(f"{label}.cooldown_ms must be positive")
    if spec.max_rate is not None and spec.max_rate <= 0:
        raise ConfigError(f"{label}.max_rate must be positive")
    if spec.direction is not None and spec.direction not in (1.0, -1.0):
        raise ConfigError(f"{label}.direction must be 1 or -1")
    if spec.swipe is not None and spec.swipe != "horizontal":
        raise ConfigError(f"{label}.swipe {spec.swipe!r} is not supported")
    if spec.alt_action is None:
        if spec.alt_max_rate is not None:
            raise ConfigError(f"{label}.alt_max_rate requires alt_action")
    else:
        if spec.alt_action not in _CONTINUOUS_ACTIONS:
            raise ConfigError(f"{label}.alt_action {spec.alt_action!r} is not supported")
        if spec.alt_max_rate is None or spec.alt_max_rate <= 0:
            raise ConfigError(f"{label}.alt_max_rate must be positive")
    if spec.still_action is None:
        if spec.still_ms is not None:
            raise ConfigError(f"{label}.still_ms requires still_action")
        return
    if spec.still_action not in _STILL_ACTIONS:
        raise ConfigError(f"{label}.still_action {spec.still_action!r} is not supported")
    if spec.still_ms is None or spec.still_ms <= 0:
        raise ConfigError(f"{label}.still_ms must be positive")


def _validate_zone(zone: Zone, label: str) -> None:
    for name, value in (
        ("x_min", zone.x_min),
        ("x_max", zone.x_max),
        ("y_min", zone.y_min),
        ("y_max", zone.y_max),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{label}.{name} must be a number")
        if value < 0.0 or value > 1.0:
            raise ConfigError(f"{label}.{name} {value} must be between 0 and 1")
    if zone.x_min >= zone.x_max or zone.y_min >= zone.y_max:
        raise ConfigError(f"{label} minimum must be less than maximum")


def _parse_zone(raw: object, label: str, fallback: Zone | None) -> Zone:
    data = _mapping(raw, label)
    _warn_unknown(label, data, {"x_min", "x_max", "y_min", "y_max"})
    if fallback is None:
        missing = {"x_min", "x_max", "y_min", "y_max"} - set(data)
        if missing:
            raise ConfigError(f"{label} missing {', '.join(sorted(missing))}")
        base = Zone(0.0, 1.0, 0.0, 1.0)
    else:
        base = fallback
    return Zone(
        _pick_float(data, "x_min", base.x_min, f"{label}.x_min"),
        _pick_float(data, "x_max", base.x_max, f"{label}.x_max"),
        _pick_float(data, "y_min", base.y_min, f"{label}.y_min"),
        _pick_float(data, "y_max", base.y_max, f"{label}.y_max"),
    )


def _per_action(raw: object) -> dict[str, bool]:
    if raw is None:
        return {}
    data = _mapping(raw, "notifications.per_action")
    parsed: dict[str, bool] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key:
            raise ConfigError("notifications.per_action keys must be non-empty strings")
        parsed[key] = _as_bool(value, f"notifications.per_action.{key}")
    return parsed


def _routing(raw: object) -> dict[str, str]:
    data = _mapping(raw, "player.routing")
    routing: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key:
            raise ConfigError("player.routing keys must be non-empty strings")
        routing[key] = _as_str(value, f"player.routing.{key}")
    return routing


def _require_positive(value: int, label: str) -> None:
    if value <= 0:
        raise ConfigError(f"{label} {value} must be positive")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ConfigError(f"{label} keys must be strings")
    return value


def _warn_unknown(section: str, data: dict[str, Any], known: set[str]) -> None:
    for key in data:
        if key not in known:
            name = f"{section}.{key}" if section else key
            warnings.warn(f"unknown config key '{name}'", ConfigWarning, stacklevel=3)


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    return float(value)


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be true or false")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{label} must be a list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_as_str(item, f"{label}[{index}]"))
    return tuple(items)


def _pick_int(data: dict[str, Any], key: str, fallback: int, label: str) -> int:
    if key not in data:
        return fallback
    return _as_int(data[key], label)


def _pick_float(data: dict[str, Any], key: str, fallback: float, label: str) -> float:
    if key not in data:
        return fallback
    return _as_float(data[key], label)


def _pick_bool(data: dict[str, Any], key: str, fallback: bool, label: str) -> bool:
    if key not in data:
        return fallback
    return _as_bool(data[key], label)


def _pick_str(data: dict[str, Any], key: str, fallback: str | None, label: str) -> str | None:
    if key not in data:
        return fallback
    return _as_str(data[key], label)


def _pick_optional_int(
    data: dict[str, Any],
    key: str,
    fallback: int | None,
    label: str,
) -> int | None:
    if key not in data:
        return fallback
    if data[key] is None:
        return None
    return _as_int(data[key], label)


def _pick_optional_float(
    data: dict[str, Any],
    key: str,
    fallback: float | None,
    label: str,
) -> float | None:
    if key not in data:
        return fallback
    if data[key] is None:
        return None
    return _as_float(data[key], label)


def _pick_optional_str(
    data: dict[str, Any],
    key: str,
    fallback: str | None,
    label: str,
) -> str | None:
    if key not in data:
        return fallback
    if data[key] is None:
        return None
    return _as_str(data[key], label)
