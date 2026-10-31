"""Replay, the action log, the process lock, and CLI parsing.

The engine-backed replay skips when that module is not in this worktree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gesture_media.action_types import Action, Kind
from gesture_media.app import (
    acquire_instance_lock,
    describe_action,
    format_action_log,
    replay,
)
from gesture_media.cli import build_parser
from gesture_media.vision.types import frame_to_json


def test_action_log_includes_gesture_and_volume_target():
    action = Action(Kind.VOLUME, "Pointing_Up", 3)
    line = format_action_log(action, volume_target="system", stamp="12:01:03")
    assert line == "12:01:03 ACTION volume +3% (system) Pointing_Up"
    assert describe_action(Action(Kind.SEEK, "Victory", -2.5), volume_target="player") == (
        "seek -2.5 s Victory"
    )
    assert describe_action(Action(Kind.PLAY_PAUSE, "Open_Palm"), volume_target="system") == (
        "play/pause Open_Palm"
    )


def test_second_lock_exits_the_contender(tmp_path, capsys):
    first = acquire_instance_lock(tmp_path)
    assert first is not None
    assert acquire_instance_lock(tmp_path) is None
    assert "already running" in capsys.readouterr().err
    first.release()
    again = acquire_instance_lock(tmp_path)
    assert again is not None
    again.release()


def test_recording_line_has_landmarks_not_pixels():
    line = frame_to_json(15, ())
    assert '"t_ms":15' in line
    assert '"hands":[]' in line
    assert "image" not in line


def test_replay_reports_missing_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "gesture_media.engine", None)
    path = tmp_path / "run.jsonl"
    path.write_text("")
    assert replay(path) == 1
    assert "engine" in capsys.readouterr().err.lower()


def test_parser_accepts_the_documented_commands():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            "config.yaml",
            "--camera",
            "1",
            "--player",
            "mpv",
            "--media",
            "a.wav",
            "b.wav",
            "--no-preview",
            "--no-calibration",
            "--verbose",
        ]
    )
    assert args.command == "run"
    assert args.camera == 1
    assert args.player == "mpv"
    assert args.media == ["a.wav", "b.wav"]
    assert args.no_preview and args.no_calibration and args.verbose
    recorded = parser.parse_args(["record", "--out", "run.jsonl"])
    assert recorded.out == Path("run.jsonl")
    replay_args = parser.parse_args(["replay", "run.jsonl"])
    assert replay_args.run == Path("run.jsonl")
    calibrate = parser.parse_args(["calibrate", "--camera", "2", "--skip-gestures", "--reset"])
    assert calibrate.camera == 2
    assert calibrate.skip_gestures and calibrate.reset
    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["list-cameras"]).command == "list-cameras"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--camera", "0", "--source", "clip.mp4"])


def test_replay_empty_recording_when_engine_is_present(tmp_path, capsys):
    try:
        import gesture_media.config  # noqa: F401
        import gesture_media.engine  # noqa: F401
        import gesture_media.players.fake  # noqa: F401
    except ImportError:
        pytest.skip("engine module is not in this worktree")
    path = tmp_path / "run.jsonl"
    path.write_text(frame_to_json(0, ()) + "\n" + frame_to_json(200, ()) + "\n")
    assert replay(path) == 0
    assert capsys.readouterr().out == ""
