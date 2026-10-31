"""In-process mpv. Pitch uses the native rubberband filter when it is present."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence

from gesture_media.players.base import Capability, pitch_scale

log = logging.getLogger(__name__)

HELP_TIMEOUT_S = 2.0
_RUBBERBAND_AF = "@rb:rubberband=pitch-scale=1"


class Unsupported(RuntimeError):
    """This backend does not implement the requested capability."""


class MpvUnavailable(RuntimeError):
    """``import mpv`` failed. Install mpv-libs (Fedora) or libmpv2 (Debian)."""


def _probe_rubberband() -> bool:
    try:
        proc = subprocess.run(
            ["mpv", "--af=help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=HELP_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("could not probe mpv filters: %s", exc)
        return False
    text = f"{proc.stdout}\n{proc.stderr}"
    return "rubberband" in text.lower()


def rubberband_available() -> bool:
    """True when ``mpv --af=help`` lists the native rubberband filter. Cached."""
    cached = getattr(rubberband_available, "cache", None)
    if cached is not None:
        return bool(cached)
    found = _probe_rubberband()
    rubberband_available.cache = found  # type: ignore[attr-defined]
    return found


rubberband_available.cache = None  # type: ignore[attr-defined]


def _scale_string(semitones: float) -> str:
    return f"{pitch_scale(semitones):.6f}"


class MpvPlayer:
    """python-mpv player and its own volume control.

    Tempo stays on the ``speed`` property. Pitch is rubberband ``set-pitch``.
    If rubberband is missing, ``vinyl_mode`` is set and ``speed`` carries the
    pitch change (tempo moves with it).
    """

    name = "mpv"
    capabilities = frozenset(Capability)

    def __init__(
        self,
        media: tuple[str, ...] = (),
        *,
        force_window: bool = True,
        ao: str | None = None,
        vo: str | None = None,
    ) -> None:
        try:
            import mpv
        except (ImportError, OSError) as exc:
            raise MpvUnavailable(
                "libmpv is not importable. Install the mpv-libs package so `import mpv` works."
            ) from exc
        options: dict[str, object] = {
            "input_default_bindings": False,
            "input_vo_keyboard": False,
            "osc": True,
            "ytdl": False,
            "idle": True,
            "force_window": force_window,
        }
        if ao is not None:
            options["ao"] = ao
        if vo is not None:
            options["vo"] = vo
        self._player = mpv.MPV(**options)
        self.media = tuple(media)
        self.pitch_semitones = 0.0
        self.vinyl_mode = not rubberband_available()
        self._filter_ready = False
        self._load(self.media)

    @classmethod
    def available(cls) -> bool:
        try:
            import mpv  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    def _load(self, media: Sequence[str]) -> None:
        for index, item in enumerate(media):
            mode = "replace" if index == 0 else "append"
            try:
                self._player.loadfile(str(item), mode)
            except Exception as exc:
                log.warning("mpv could not load %s (%s)", item, exc)

    def is_available(self) -> bool:
        return self._player is not None

    def _guard(self, label: str, fn) -> None:
        if self._player is None:
            return
        try:
            fn()
        except Exception as exc:
            log.warning("mpv %s failed: %s", label, exc)

    def play_pause(self) -> None:
        def toggle() -> None:
            self._player.pause = not bool(self._player.pause)

        self._guard("play_pause", toggle)

    def next(self) -> None:
        self._guard("next", lambda: self._player.command("playlist-next"))

    def previous(self) -> None:
        self._guard("previous", lambda: self._player.command("playlist-prev"))

    def seek(self, seconds: float) -> None:
        self._guard("seek", lambda: self._player.command("seek", float(seconds), "relative"))

    def _use_rubberband(self) -> bool:
        if self.vinyl_mode or self._player is None:
            return False
        if self._filter_ready:
            return True
        try:
            self._player.af = _RUBBERBAND_AF
        except Exception as exc:
            log.warning("rubberband filter was rejected (%s); using vinyl mode", exc)
            self.vinyl_mode = True
            return False
        self._filter_ready = True
        self.vinyl_mode = False
        return True

    def set_pitch(self, semitones: float) -> None:
        self.pitch_semitones = float(semitones)

        def apply() -> None:
            if self._use_rubberband():
                self._player.speed = 1
                self._player.command("af-command", "rb", "set-pitch", _scale_string(semitones))
                self._player.speed = 1
                return
            self.vinyl_mode = True
            self._player.speed = pitch_scale(semitones)

        self._guard("set_pitch", apply)

    def open(self) -> None:
        def raise_window() -> None:
            self._player.force_window = True

        self._guard("open", raise_window)

    def close(self) -> None:
        player = self._player
        self._player = None
        if player is None:
            return
        try:
            player.terminate()
        except Exception as exc:
            log.warning("mpv terminate failed: %s", exc)

    def get_position(self) -> float | None:
        if self._player is None:
            return None
        try:
            value = self._player.time_pos
        except Exception as exc:
            log.warning("mpv position failed: %s", exc)
            return None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def volume_delta(self, percent: float) -> None:
        def apply() -> None:
            current = float(self._player.volume or 0.0)
            self._player.volume = max(0.0, min(100.0, current + float(percent)))

        self._guard("volume", apply)

    def toggle_mute(self) -> None:
        def apply() -> None:
            self._player.mute = not bool(self._player.mute)

        self._guard("mute", apply)

    def get_state(self) -> tuple[float, bool]:
        if self._player is None:
            return (0.0, False)
        try:
            volume = float(self._player.volume or 0.0)
            muted = bool(self._player.mute)
        except Exception as exc:
            log.warning("mpv volume state failed: %s", exc)
            return (0.0, False)
        return volume, muted
