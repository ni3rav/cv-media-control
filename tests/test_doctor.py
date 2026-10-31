"""Doctor decisions from a snapshot. These tests do not open a camera."""

from __future__ import annotations

import pytest

from gesture_media.doctor import DoctorEnv, evaluate, run_doctor


def _env(**overrides) -> DoctorEnv:
    values = dict(
        video_devices=("/dev/video0",),
        camera_open_ok=True,
        model_exists=True,
        model_path="models/gesture_recognizer.task",
        playerctl=True,
        wpctl=True,
        mpv_import=True,
        rubberband=True,
        players=("vlc",),
        session_type="wayland",
        desktop="KDE",
        profile_version=1,
        profile_error="",
        tray_ok=True,
        tray_detail="pystray is installed and a display is set",
        notify_ok=True,
        notify_detail="org.freedesktop.Notifications is owned",
        mediapipe=True,
    )
    values.update(overrides)
    return DoctorEnv(**values)


def _by_name(checks, name: str):
    return next(check for check in checks if check.name == name)


def test_healthy_snapshot_exits_zero():
    code, checks = evaluate(_env())
    assert code == 0
    assert _by_name(checks, "model").hard is True
    assert _by_name(checks, "model").status == "PASS"
    assert "HARD" in _by_name(checks, "playback").line()


def test_missing_model_is_hard():
    code, checks = evaluate(_env(model_exists=False))
    assert code == 1
    model = _by_name(checks, "model")
    assert model.status == "FAIL"
    assert model.hard is True
    assert "HARD" in model.line()


def test_mediapipe_missing_is_hard():
    code, checks = evaluate(_env(mediapipe=False))
    assert code == 1
    assert _by_name(checks, "mediapipe").status == "FAIL"
    assert _by_name(checks, "mediapipe").hard is True


def test_no_playback_backend_is_hard():
    code, checks = evaluate(_env(playerctl=False, mpv_import=False, players=()))
    assert code == 1
    playback = _by_name(checks, "playback")
    assert playback.status == "FAIL"
    assert playback.hard is True
    assert "neither" in playback.detail


def test_playerctl_alone_is_enough():
    code, checks = evaluate(_env(playerctl=True, mpv_import=False))
    assert code == 0
    assert _by_name(checks, "libmpv").status == "FAIL"
    assert _by_name(checks, "libmpv").hard is False
    assert _by_name(checks, "playback").status == "PASS"


def test_libmpv_alone_is_enough():
    code, checks = evaluate(_env(playerctl=False, mpv_import=True, players=()))
    assert code == 0
    assert _by_name(checks, "playerctl").hard is False
    assert _by_name(checks, "playback").status == "PASS"


def test_camera_open_failure_is_soft():
    code, checks = evaluate(_env(camera_open_ok=False))
    assert code == 0
    camera = _by_name(checks, "camera")
    assert camera.status == "FAIL"
    assert camera.hard is False
    assert "soft" in camera.detail


def test_missing_camera_device_is_a_warning():
    code, checks = evaluate(_env(video_devices=(), camera_open_ok=False))
    assert code == 0
    assert _by_name(checks, "camera").status == "WARN"


def test_rubberband_missing_is_soft():
    code, checks = evaluate(_env(rubberband=False))
    assert code == 0
    rubber = _by_name(checks, "rubberband")
    assert rubber.status == "FAIL"
    assert rubber.hard is False
    assert "vinyl" in rubber.detail


def test_profile_and_session_are_reported():
    code, checks = evaluate(
        _env(profile_version=None, profile_error="", session_type="", desktop="")
    )
    assert code == 0
    assert _by_name(checks, "calibration").status == "WARN"
    assert "optional" in _by_name(checks, "calibration").detail
    assert _by_name(checks, "session").status == "WARN"
    present, present_checks = evaluate(_env(profile_version=1))
    assert present == 0
    assert "version 1" in _by_name(present_checks, "calibration").detail


def test_run_doctor_prints_hard_lines_without_probing(monkeypatch, capsys):
    def boom():
        raise AssertionError("collect_environment must not run")

    monkeypatch.setattr("gesture_media.doctor.collect_environment", boom)
    code = run_doctor(env=_env(model_exists=False, mediapipe=False))
    assert code == 1
    out = capsys.readouterr().out
    assert "HARD" in out
    assert "model" in out
    assert "doctor: hard failure" in out


def test_run_doctor_success_line(monkeypatch, capsys):
    monkeypatch.setattr(
        "gesture_media.doctor.collect_environment",
        lambda: pytest.fail("probed the machine"),
    )
    assert run_doctor(env=_env()) == 0
    assert "doctor: no hard failures" in capsys.readouterr().out
