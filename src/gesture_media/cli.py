"""Command line entry point for gesture-media."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gesture-media")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="watch the camera and control media")
    _add_run_flags(run)

    sub.add_parser("doctor", help="check the camera, model, players, and desktop services")
    sub.add_parser("list-cameras", help="print camera indexes that open")

    record = sub.add_parser("record", help="run and write landmarks to JSONL")
    record.add_argument(
        "--out",
        required=True,
        type=Path,
        help="JSONL path for landmarks and labels",
    )
    _add_run_flags(record)

    replay = sub.add_parser("replay", help="play a JSONL recording through the engine")
    replay.add_argument("run", type=Path, help="recording produced by record")
    replay.add_argument("--config", type=Path, default=None)

    calibrate = sub.add_parser("calibrate", help="optionally tune thresholds for this camera")
    calibrate.add_argument("--config", type=Path, default=None)
    calibrate.add_argument("--camera", type=int, default=None)
    calibrate.add_argument("--skip-gestures", action="store_true")
    calibrate.add_argument("--reset", action="store_true")
    return parser


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=None)
    source.add_argument("--source", type=Path, default=None)
    parser.add_argument("--player", choices=("auto", "mpv", "mpris"), default=None)
    parser.add_argument("--media", nargs="+", default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _overrides(args: argparse.Namespace):
    from gesture_media.config import CliOverrides

    media = tuple(args.media) if getattr(args, "media", None) else None
    source = getattr(args, "source", None)
    return CliOverrides(
        camera_index=getattr(args, "camera", None),
        source=str(source) if source else None,
        player_backend=getattr(args, "player", None),
        media=media,
        no_preview=bool(getattr(args, "no_preview", False)),
        no_calibration=bool(getattr(args, "no_calibration", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _load(args: argparse.Namespace):
    from gesture_media.config import ConfigError, load_config

    try:
        return load_config(
            getattr(args, "config", None),
            use_profile=not bool(getattr(args, "no_calibration", False)),
            cli=_overrides(args),
        )
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return None


def _maybe_hint(no_calibration: bool) -> None:
    if no_calibration:
        return
    from gesture_media.config import profile_path

    if not profile_path().exists():
        print("No calibration profile yet. `gesture-media calibrate` is optional.")


def _run_command(args: argparse.Namespace, record_path: Path | None) -> int:
    from gesture_media.app import acquire_instance_lock
    from gesture_media.config import runtime_dir

    _maybe_hint(bool(getattr(args, "no_calibration", False)))
    lock = acquire_instance_lock(runtime_dir())
    if lock is None:
        return 1
    try:
        config = _load(args)
        if config is None:
            return 1
        from gesture_media.app import run

        return run(config, config_path=getattr(args, "config", None), record_path=record_path)
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(bool(getattr(args, "verbose", False)))
    try:
        if args.command == "doctor":
            from gesture_media.doctor import run_doctor

            return run_doctor()
        if args.command == "list-cameras":
            from gesture_media.app import list_cameras

            return list_cameras()
        if args.command == "replay":
            from gesture_media.app import replay
            from gesture_media.config import ConfigError, load_config

            try:
                config = load_config(args.config, use_profile=True)
            except ConfigError as exc:
                print(exc, file=sys.stderr)
                return 1
            except ImportError as exc:
                print(f"replay needs the config module ({exc}).", file=sys.stderr)
                return 1
            return replay(args.run, config)
        if args.command == "calibrate":
            from gesture_media.calibrate_ui import run_calibration
            from gesture_media.config import ConfigError, load_config

            if args.reset:
                return run_calibration(None, reset=True)
            from gesture_media.app import acquire_instance_lock
            from gesture_media.config import runtime_dir

            lock = acquire_instance_lock(runtime_dir())
            if lock is None:
                return 1
            try:
                try:
                    config = load_config(args.config, use_profile=True)
                except ConfigError as exc:
                    print(exc, file=sys.stderr)
                    return 1
                return run_calibration(
                    config,
                    camera=args.camera,
                    skip_gestures=args.skip_gestures,
                    reset=False,
                )
            finally:
                lock.release()
        if args.command == "record":
            return _run_command(args, args.out)
        if args.command == "run":
            return _run_command(args, None)
    except ImportError as exc:
        print(f"Missing module: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
