"""Build the player and audio backends selected by config.

Adding a backend is a new entry in ``FACTORIES``. The engine and the
dispatcher do not change.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable

from gesture_media.players.audio_wpctl import WpctlAudio
from gesture_media.players.base import AudioControl, Capability, Player
from gesture_media.players.mpris import MprisPlayer
from gesture_media.players.mpv import MpvPlayer

log = logging.getLogger(__name__)

COMMAND_TIMEOUT_S = 1.0


class _MissingPlayer:
    """No-op player used when the selected backend cannot be constructed."""

    capabilities: frozenset[Capability] = frozenset()

    def __init__(self, name: str) -> None:
        self.name = name
        self._warned = False

    def _warn(self) -> None:
        if self._warned:
            return
        self._warned = True
        log.warning("player backend %s is not available", self.name)

    def is_available(self) -> bool:
        return False

    def play_pause(self) -> None:
        self._warn()

    def next(self) -> None:
        self._warn()

    def previous(self) -> None:
        self._warn()

    def seek(self, seconds: float) -> None:
        del seconds
        self._warn()

    def set_pitch(self, semitones: float) -> None:
        del semitones
        self._warn()

    def open(self) -> None:
        self._warn()

    def close(self) -> None:
        return None

    def get_position(self) -> float | None:
        return None


def _playerctl(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("playerctl %s failed (%s)", argv, exc)
        return None


def _listed_players() -> list[str]:
    proc = _playerctl(["playerctl", "-l"])
    if proc is None or proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _player_status(name: str) -> str:
    proc = _playerctl(["playerctl", "-p", name, "status"])
    if proc is None or proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _players_listed() -> bool:
    return bool(_listed_players())


def best_mpris_player() -> str | None:
    """Prefer a Playing player, then Paused, then Stopped.

    Browsers stay in ``playerctl -l`` after the tab has stopped, and
    play-pause then fails. A player that is actually playing wins.
    """
    ranked: list[tuple[int, str, str]] = []
    for name in _listed_players():
        status = _player_status(name)
        rank = {"Playing": 0, "Paused": 1, "Stopped": 2}.get(status, 3)
        ranked.append((rank, name, status or "unknown"))
    if not ranked:
        return None
    ranked.sort()
    _rank, name, status = ranked[0]
    if status == "Stopped":
        log.warning(
            "%s is registered but nothing is playing. Start a song or video there, "
            "or rerun with --player mpv --media FILE",
            name,
        )
    else:
        log.info("controlling %s (%s)", name, status)
    return name


def _build_mpris(config) -> MprisPlayer:
    command = tuple(getattr(config.player, "open_command", ()) or ("vlc",))
    explicit = getattr(config.player, "mpris_player", None)
    return MprisPlayer(explicit or best_mpris_player(), command)


def _build_mpv(config) -> MpvPlayer:
    media = tuple(getattr(config.player, "media", ()) or ())
    return MpvPlayer(media, force_window=True)


FACTORIES: dict[str, Callable] = {
    "mpris": _build_mpris,
    "mpv": _build_mpv,
}


def _volume_target(config, backend_name: str) -> str:
    from gesture_media.config import resolve_volume_target

    return resolve_volume_target(config, backend_name)


def _implements_audio(obj: object) -> bool:
    names = ("volume_delta", "toggle_mute", "get_state")
    return all(callable(getattr(obj, name, None)) for name in names)


class RoutedPlayer:
    """Player facade. ``routing`` sends a capability to a named backend."""

    def __init__(
        self,
        name: str,
        players: dict[str, Player],
        routing: dict[str, str],
    ) -> None:
        self.name = name
        self.players = players
        self.routing = dict(routing)
        selected = players[name]
        self.capabilities = selected.capabilities

    def routed_player(self, capability: Capability | str) -> Player:
        key = capability.value if isinstance(capability, Capability) else str(capability)
        target_name = self.routing.get(key, self.name)
        player = self.players.get(target_name)
        if player is None:
            log.warning("no backend %s for %s; using %s", target_name, key, self.name)
            return self.players[self.name]
        return player

    def is_available(self) -> bool:
        return self.players[self.name].is_available()

    def play_pause(self) -> None:
        self.routed_player(Capability.PLAY_PAUSE).play_pause()

    def next(self) -> None:
        self.routed_player(Capability.NEXT).next()

    def previous(self) -> None:
        self.routed_player(Capability.PREVIOUS).previous()

    def seek(self, seconds: float) -> None:
        self.routed_player(Capability.SEEK).seek(seconds)

    def set_pitch(self, semitones: float) -> None:
        self.routed_player(Capability.PITCH).set_pitch(semitones)

    def open(self) -> None:
        self.routed_player(Capability.OPEN).open()

    def close(self) -> None:
        seen: set[int] = set()
        for player in self.players.values():
            identity = id(player)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                player.close()
            except Exception as exc:
                log.warning("closing %s failed: %s", getattr(player, "name", player), exc)

    def get_position(self) -> float | None:
        return self.routed_player(Capability.SEEK).get_position()

    @property
    def vinyl_mode(self) -> bool:
        player = self.routed_player(Capability.PITCH)
        return bool(getattr(player, "vinyl_mode", False))


def _resolve_backend_name(config) -> str:
    backend = config.player.backend
    if backend == "auto":
        return "mpris" if _players_listed() else "mpv"
    return str(backend)


def _select_audio(config, backend_name: str, players: dict[str, Player]) -> AudioControl:
    target = _volume_target(config, backend_name)
    selected = players.get(backend_name)
    if target == "player" and _implements_audio(selected):
        return selected  # type: ignore[return-value]
    if target == "player":
        log.info("player volume is unavailable on %s; using wpctl", backend_name)
    return WpctlAudio()


def build_backends(config) -> tuple[Player, AudioControl, str]:
    """Return ``(player, audio, backend_name)``.

    ``auto`` uses MPRIS when ``playerctl -l`` lists a player, otherwise mpv.
    System volume is ``WpctlAudio``. Player volume uses the mpv object when
    that object implements audio. ``config.player.routing`` maps a capability
    name to a backend name; an empty map keeps every transport command on the
    selected player.
    """
    name = _resolve_backend_name(config)
    routing = {str(key): str(value) for key, value in dict(config.player.routing or {}).items()}
    needed = {name, *routing.values()}
    players: dict[str, Player] = {}
    for key in sorted(needed):
        factory = FACTORIES.get(key)
        if factory is None:
            log.warning("unknown player backend %s", key)
            continue
        try:
            players[key] = factory(config)
        except Exception as exc:
            log.warning("backend %s is unavailable: %s", key, exc)
    if name not in players:
        log.error("selected backend %s could not be started", name)
        players[name] = _MissingPlayer(name)
    audio = _select_audio(config, name, players)
    return RoutedPlayer(name, players, routing), audio, name
