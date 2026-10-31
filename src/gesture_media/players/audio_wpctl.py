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

    def volume_delta(self, percent: float) -> None:
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
        _run(["wpctl", "set-mute", _SINK, "toggle"])

    def get_state(self) -> tuple[float, bool]:
        proc = _run(["wpctl", "get-volume", _SINK])
        if proc is None or proc.returncode != 0:
            return (0.0, False)
        try:
            return parse_wpctl_volume(proc.stdout)
        except ValueError as exc:
            log.warning("%s", exc)
            return (0.0, False)
