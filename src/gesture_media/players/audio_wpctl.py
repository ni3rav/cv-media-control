"""System volume through WirePlumber's wpctl. Failures are logged, not raised."""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Sequence

log = logging.getLogger(__name__)

COMMAND_TIMEOUT_S = 1.0
_SINK = "@DEFAULT_AUDIO_SINK@"
_VOLUME_RE = re.compile(r"Volume:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_MUTED_RE = re.compile(r"\[MUTED\]", re.IGNORECASE)


class Unsupported(RuntimeError):
    """This backend does not implement the requested capability."""


def format_volume_step(percent: float) -> str:
    """wpctl volume delta, for example ``5%+`` or ``3%-``."""
    magnitude = abs(float(percent))
    if magnitude.is_integer():
        text = str(int(magnitude))
    else:
        text = f"{magnitude:.3f}".rstrip("0").rstrip(".")
    sign = "+" if percent >= 0 else "-"
    return f"{text}%{sign}"


def parse_wpctl_volume(text: str) -> tuple[float, bool]:
    """Parse ``wpctl get-volume`` into ``(percent 0-100, muted)``."""
    match = _VOLUME_RE.search(text)
    if match is None:
        raise ValueError(f"unrecognized wpctl volume output: {text!r}")
    percent = round(float(match.group(1)) * 100.0, 2)
    return percent, _MUTED_RE.search(text) is not None


def playback_stream_ids(text: str) -> list[str]:
    """Playback stream node ids from ``wpctl status``.

    Link lines contain ``>`` and are ports, not streams. Sources are outside
    this section, so the microphone is left alone.
    """
    ids: list[str] = []
    capturing = False
    for line in text.splitlines():
        if not capturing:
            if "Streams:" in line:
                capturing = True
            continue
        if line.startswith(("Video", "Settings")):
            break
        if ">" in line:
            continue
        match = re.search(r"(\d+)\.\s+\S", line)
        if match:
            ids.append(match.group(1))
    return ids


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("wpctl failed: %s (%s)", list(argv), exc)
        return None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        log.warning("wpctl %s exited %s (%s)", list(argv), completed.returncode, detail)
    return completed


class WpctlAudio:
    """``@DEFAULT_AUDIO_SINK@`` volume and mute. The limiter is ``-l 1.0``."""

    def __init__(self) -> None:
        # The first volume-up checks mute. After that, only a mute we applied
        # makes the next volume-up look again.
        self._check_mute_on_volume_up = True

    def volume_delta(self, percent: float) -> None:
        if percent > 0 and self._check_mute_on_volume_up:
            muted = self._read_muted()
            if muted is True:
                self._set_muted(False)
            elif muted is False:
                self._check_mute_on_volume_up = False
        _run(
            [
                "wpctl",
                "set-volume",
                _SINK,
                format_volume_step(percent),
                "-l",
                "1.0",
            ]
        )

    def toggle_mute(self) -> None:
        """Mute the sink, or unmute the sink and any muted playback stream.

        ``toggle`` on the sink alone leaves a browser stream muted, so the
        music stays silent. A failed volume read falls back to ``toggle``.
        """
        muted = self._read_muted()
        if muted is None:
            _run(["wpctl", "set-mute", _SINK, "toggle"])
            self._check_mute_on_volume_up = True
            return
        if muted:
            self._set_muted(False)
            return
        streams = self._muted_stream_ids()
        if streams:
            for stream_id in streams:
                _run(["wpctl", "set-mute", stream_id, "0"])
            self._check_mute_on_volume_up = False
            return
        self._set_muted(True)

    def _read_muted(self) -> bool | None:
        proc = _run(["wpctl", "get-volume", _SINK])
        if proc is None or proc.returncode != 0:
            return None
        try:
            _percent, muted = parse_wpctl_volume(proc.stdout)
        except ValueError:
            return None
        return muted

    def _stream_ids(self) -> list[str]:
        proc = _run(["wpctl", "status"])
        if proc is None or proc.returncode != 0:
            return []
        return playback_stream_ids(proc.stdout)

    def _muted_stream_ids(self) -> list[str]:
        muted: list[str] = []
        for stream_id in self._stream_ids():
            proc = _run(["wpctl", "get-volume", stream_id])
            if proc is None or proc.returncode != 0:
                continue
            try:
                _percent, stream_muted = parse_wpctl_volume(proc.stdout)
            except ValueError:
                continue
            if stream_muted:
                muted.append(stream_id)
        return muted

    def _set_muted(self, muted: bool) -> None:
        _run(["wpctl", "set-mute", _SINK, "1" if muted else "0"])
        if not muted:
            for stream_id in self._stream_ids():
                _run(["wpctl", "set-mute", stream_id, "0"])
        self._check_mute_on_volume_up = muted

    def get_state(self) -> tuple[float, bool]:
        proc = _run(["wpctl", "get-volume", _SINK])
        if proc is None or proc.returncode != 0:
            return (0.0, False)
        try:
            return parse_wpctl_volume(proc.stdout)
        except ValueError as exc:
            log.warning("%s", exc)
            return (0.0, False)
