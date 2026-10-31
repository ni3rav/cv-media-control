"""MPRIS backend. Talks to playerctl and never raises into the frame loop."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence

from gesture_media.players.base import Capability

log = logging.getLogger(__name__)

COMMAND_TIMEOUT_S = 1.0


class Unsupported(RuntimeError):
    """This backend does not implement the requested capability."""


def format_relative_position(seconds: float) -> str:
    """playerctl position argument, for example ``5+`` or ``5-``."""
    magnitude = abs(float(seconds))
    if magnitude.is_integer():
        text = str(int(magnitude))
    else:
        text = f"{magnitude:.3f}".rstrip("0").rstrip(".")
    sign = "+" if seconds >= 0 else "-"
    return f"{text}{sign}"


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("playerctl failed: %s (%s)", list(argv), exc)
        return None


class MprisPlayer:
    """playerctl transport. Pitch and volume are not MPRIS capabilities here."""

    name = "mpris"
    capabilities = frozenset(
        {
            Capability.PLAY_PAUSE,
            Capability.NEXT,
            Capability.PREVIOUS,
            Capability.SEEK,
            Capability.OPEN,
        }
    )

    def __init__(self, player_name: str | None, open_command: tuple[str, ...]) -> None:
        self.player_name = player_name or None
        self.open_command = tuple(open_command)
        self._warned: set[Capability] = set()

    def _argv(self, *args: str) -> list[str]:
        argv = ["playerctl"]
        if self.player_name:
            argv.extend(["--player", self.player_name])
        argv.extend(args)
        return argv

    def _unsupported(self, capability: Capability) -> None:
        if capability in self._warned:
            return
        self._warned.add(capability)
        log.warning("%s is not supported by %s", capability.value, self.name)

    def _invoke(self, *args: str) -> subprocess.CompletedProcess[str] | None:
        argv = self._argv(*args)
        proc = _run(argv)
        if proc is not None and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            log.warning("playerctl %s exited %s (%s)", list(argv), proc.returncode, detail)
        return proc

    def is_available(self) -> bool:
        proc = _run(self._argv("-l"))
        if proc is None or proc.returncode != 0:
            return False
        return any(line.strip() for line in proc.stdout.splitlines())

    def play_pause(self) -> None:
        self._invoke("play-pause")

    def next(self) -> None:
        self._invoke("next")

    def previous(self) -> None:
        self._invoke("previous")

    def seek(self, seconds: float) -> None:
        self._invoke("position", format_relative_position(seconds))

    def set_pitch(self, semitones: float) -> None:
        del semitones
        self._unsupported(Capability.PITCH)

    def open(self) -> None:
        if not self.open_command:
            log.warning("mpris open_command is empty")
            return
        try:
            subprocess.Popen(
                list(self.open_command),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            log.warning("could not launch %s (%s)", list(self.open_command), exc)

    def close(self) -> None:
        return None

    def get_position(self) -> float | None:
        proc = self._invoke("position")
        if proc is None or proc.returncode != 0:
            return None
        token = proc.stdout.strip().split()
        if not token:
            return None
        try:
            return float(token[0])
        except ValueError:
            log.warning("playerctl position was not a number: %s", proc.stdout.strip())
            return None
