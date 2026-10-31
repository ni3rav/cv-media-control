"""Environment report. Hard failures exit 1. Everything else is printed and ignored."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_MODEL_NAME = "gesture_recognizer.task"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    hard: bool
    detail: str

    def line(self) -> str:
        tag = "HARD" if self.hard else "soft"
        return f"{self.status:4} [{tag}] {self.name}: {self.detail}"


@dataclass(frozen=True)
class DoctorEnv:
    video_devices: tuple[str, ...]
    camera_open_ok: bool
    model_exists: bool
    model_path: str
    playerctl: bool
    wpctl: bool
    mpv_import: bool
    rubberband: bool | None
    players: tuple[str, ...]
    session_type: str
    desktop: str
    profile_version: int | None
    profile_error: str
    tray_ok: bool
    tray_detail: str
    notify_ok: bool
    notify_detail: str
    mediapipe: bool


def model_path() -> Path:
    cwd_candidate = Path.cwd() / "models" / _MODEL_NAME
    if cwd_candidate.is_file():
        return cwd_candidate
    return Path(__file__).resolve().parents[2] / "models" / _MODEL_NAME


def evaluate(env: DoctorEnv) -> tuple[int, tuple[Check, ...]]:
    """Turn a snapshot into checks. A hard FAIL makes the exit code 1."""
    checks: list[Check] = []
    if not env.video_devices:
        checks.append(Check("camera", "WARN", False, "no /dev/video* devices"))
    elif env.camera_open_ok:
        checks.append(Check("camera", "PASS", False, f"opened {env.video_devices[0]}"))
    else:
        checks.append(
            Check(
                "camera",
                "FAIL",
                False,
                "device exists but open failed (soft; a headless run can still exit 0)",
            )
        )
    if env.model_exists:
        checks.append(Check("model", "PASS", True, f"found {env.model_path}"))
    else:
        checks.append(Check("model", "FAIL", True, f"missing {env.model_path}"))
    if env.mediapipe:
        checks.append(Check("mediapipe", "PASS", True, "import ok"))
    else:
        checks.append(Check("mediapipe", "FAIL", True, "cannot import mediapipe"))
    checks.append(
        Check(
            "playerctl",
            "PASS" if env.playerctl else "FAIL",
            False,
            "on PATH" if env.playerctl else "not on PATH (install playerctl)",
        )
    )
    checks.append(
        Check(
            "wpctl",
            "PASS" if env.wpctl else "FAIL",
            False,
            "on PATH" if env.wpctl else "not on PATH (install wireplumber)",
        )
    )
    checks.append(
        Check(
            "libmpv",
            "PASS" if env.mpv_import else "FAIL",
            False,
            (
                "import mpv ok"
                if env.mpv_import
                else "import mpv failed (install mpv-libs or libmpv2)"
            ),
        )
    )
    playback_ok = env.playerctl or env.mpv_import
    checks.append(
        Check(
            "playback",
            "PASS" if playback_ok else "FAIL",
            True,
            "playerctl or libmpv is available"
            if playback_ok
            else "neither playerctl nor libmpv is available",
        )
    )
    if env.rubberband is None:
        checks.append(Check("rubberband", "WARN", False, "mpv --af=help could not be run"))
    elif env.rubberband:
        checks.append(Check("rubberband", "PASS", False, "mpv --af=help lists rubberband"))
    else:
        checks.append(
            Check(
                "rubberband",
                "FAIL",
                False,
                "rubberband filter is missing; pitch uses vinyl mode",
            )
        )
    if not env.playerctl:
        checks.append(Check("players", "WARN", False, "playerctl is not available"))
    elif env.players:
        checks.append(Check("players", "PASS", False, "active: " + ", ".join(env.players)))
    else:
        checks.append(Check("players", "WARN", False, "no active MPRIS players"))
    session_type = env.session_type or "unset"
    desktop = env.desktop or "unset"
    session = f"XDG_SESSION_TYPE={session_type} XDG_CURRENT_DESKTOP={desktop}"
    checks.append(
        Check(
            "session",
            "PASS" if env.session_type and env.desktop else "WARN",
            False,
            session,
        )
    )
    if env.profile_error:
        checks.append(Check("calibration", "FAIL", False, env.profile_error))
    elif env.profile_version is None:
        checks.append(Check("calibration", "WARN", False, "no calibration profile (optional)"))
    else:
        checks.append(Check("calibration", "PASS", False, f"profile version {env.profile_version}"))
    checks.append(
        Check("tray", "PASS" if env.tray_ok else "WARN", False, env.tray_detail),
    )
    checks.append(
        Check(
            "notifications",
            "PASS" if env.notify_ok else "WARN",
            False,
            env.notify_detail,
        )
    )
    hard_fail = any(check.hard and check.status == "FAIL" for check in checks)
    return (1 if hard_fail else 0), tuple(checks)


def _probe_camera(devices: tuple[str, ...]) -> bool:
    if not devices:
        return False
    try:
        import cv2
    except ImportError as exc:
        log.info("camera probe skipped: %s", exc)
        return False
    capture = cv2.VideoCapture(0)
    try:
        return bool(capture.isOpened())
    finally:
        capture.release()


def _probe_profile() -> tuple[int | None, str]:
    try:
        from gesture_media.config import PROFILE_VERSION, load_profile, profile_path
    except ImportError:
        return None, "config module is not available yet"
    path = profile_path()
    if not path.exists():
        return None, ""
    profile = load_profile(path)
    if profile is None:
        return None, f"profile at {path} is missing, corrupt, or an old version"
    version = getattr(profile, "version", None)
    if version != PROFILE_VERSION:
        found = version if isinstance(version, int) else None
        return found, f"unexpected profile version {version}"
    return int(version), ""


def _active_players() -> tuple[str, ...]:
    if shutil.which("playerctl") is None:
        return ()
    try:
        proc = subprocess.run(
            ["playerctl", "-l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("playerctl -l failed: %s", exc)
        return ()
    return tuple(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _mpv_imports() -> bool:
    try:
        import mpv  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _mediapipe_imports() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return True


def collect_environment() -> DoctorEnv:
    devices = tuple(sorted(str(path) for path in Path("/dev").glob("video*")))
    from gesture_media.notify import notification_service_available
    from gesture_media.players.mpv import rubberband_available
    from gesture_media.tray import tray_available

    notify_ok, notify_detail = notification_service_available()
    tray_ok, tray_detail = tray_available()
    try:
        rubberband: bool | None = rubberband_available()
    except Exception as exc:
        log.info("rubberband probe failed: %s", exc)
        rubberband = None
    version, profile_error = _probe_profile()
    located = model_path()
    return DoctorEnv(
        video_devices=devices,
        camera_open_ok=_probe_camera(devices),
        model_exists=located.is_file(),
        model_path=str(located),
        playerctl=shutil.which("playerctl") is not None,
        wpctl=shutil.which("wpctl") is not None,
        mpv_import=_mpv_imports(),
        rubberband=rubberband,
        players=_active_players(),
        session_type=os.environ.get("XDG_SESSION_TYPE", ""),
        desktop=os.environ.get("XDG_CURRENT_DESKTOP", ""),
        profile_version=version,
        profile_error=profile_error,
        tray_ok=tray_ok,
        tray_detail=tray_detail,
        notify_ok=notify_ok,
        notify_detail=notify_detail,
        mediapipe=_mediapipe_imports(),
    )


def run_doctor(env: DoctorEnv | None = None) -> int:
    snapshot = collect_environment() if env is None else env
    code, checks = evaluate(snapshot)
    for check in checks:
        print(check.line())
    if code:
        print("doctor: hard failure")
    else:
        print("doctor: no hard failures")
    return code
