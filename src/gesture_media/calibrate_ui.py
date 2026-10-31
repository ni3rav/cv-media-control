"""Guided calibration. Esc cancels and does not write a profile.

The preview window stays on the main thread. ``y`` and ``n`` are read from
that window and, while it is open, from stdin as well. Without a window the
question is stdin only.
"""

from __future__ import annotations

import logging
import select
import sys
import time

log = logging.getLogger(__name__)

WINDOW = "Gesture Media Calibration"
_GESTURES = (
    "Open_Palm",
    "Closed_Fist",
    "Thumb_Up",
    "Thumb_Down",
    "Pointing_Up",
    "Victory",
    "ILoveYou",
)


def run_calibration(
    config,
    *,
    camera: int | None = None,
    skip_gestures: bool = False,
    reset: bool = False,
) -> int:
    from gesture_media.config import profile_path

    if reset:
        path = profile_path()
        if path.exists():
            path.unlink()
            print(f"Removed {path}")
        else:
            print(f"No calibration profile at {path}")
        return 0
    try:
        from gesture_media.sources import CameraOpenError, bgr_to_rgb, mirror_bgr, open_source
        from gesture_media.vision.types import palm_center
    except ImportError as exc:
        print(f"calibrate needs the camera source ({exc}).", file=sys.stderr)
        return 1
    active = _with_camera(config, camera)
    try:
        source = open_source(active)
    except CameraOpenError as exc:
        print(f"Could not open the camera: {exc}", file=sys.stderr)
        return 2
    classifier = None
    preview = False
    try:
        classifier = _open_classifier(active)
        preview = _try_window()
        samples, scores, cancelled = _collect_phases(
            source,
            classifier,
            palm_center,
            mirror_bgr,
            bgr_to_rgb,
            skip_gestures=skip_gestures,
            preview=preview,
        )
        if cancelled:
            print("Calibration cancelled")
            _notify(active, "cancelled")
            return 0
        saved = _summarize_and_maybe_save(active, samples, scores, preview=preview)
        _notify(active, "saved" if saved else "cancelled")
        if not saved:
            print("Calibration cancelled")
        return 0
    except KeyboardInterrupt:
        print("Calibration cancelled")
        _notify(active, "cancelled")
        return 0
    except (ImportError, FileNotFoundError) as exc:
        print(f"Calibration cannot start: {exc}", file=sys.stderr)
        return 1
    finally:
        if classifier is not None:
            try:
                classifier.close()
            except Exception:
                log.debug("classifier close failed", exc_info=True)
        try:
            source.release()
        except Exception:
            log.debug("camera release failed", exc_info=True)
        if preview:
            _destroy_window()


def _with_camera(config, camera: int | None):
    if camera is None:
        return config
    from dataclasses import replace

    return replace(config, camera=replace(config.camera, index=camera))


def _open_classifier(config):
    if config.vision.classifier == "rules":
        from gesture_media.vision.rules import RulesClassifier

        return RulesClassifier()
    from gesture_media.app import model_path
    from gesture_media.vision.classifier import MediaPipeClassifier

    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model file is missing: {path}")
    return MediaPipeClassifier(path, num_hands=1, min_score=config.vision.min_score)


def _phases(skip_gestures: bool) -> tuple[tuple[str, int, str], ...]:
    phases = [
        ("rest", 3000, "Hold a relaxed hand still"),
        ("top", 1000, "Move your palm to a comfortable top and hold"),
        ("bottom", 1000, "Move your palm to a comfortable bottom and hold"),
        ("left", 1000, "Move your palm to a comfortable left and hold"),
        ("right", 1000, "Move your palm to a comfortable right and hold"),
    ]
    if not skip_gestures:
        for name in _GESTURES:
            phases.append((f"gesture:{name}", 2000, f"Show {name.replace('_', ' ')}"))
    return tuple(phases)


def _collect_phases(
    source,
    classifier,
    palm_center,
    mirror_bgr,
    bgr_to_rgb,
    *,
    skip_gestures: bool,
    preview: bool,
):
    from gesture_media.calibration import PalmSample

    buckets: dict[str, list] = {name: [] for name in ("rest", "top", "bottom", "left", "right")}
    scores: dict[str, list[float]] = {}
    origin = time.monotonic()
    last_t = -1
    for key, duration_ms, prompt in _phases(skip_gestures):
        print(prompt)
        deadline = time.monotonic() + duration_ms / 1000
        while time.monotonic() < deadline:
            if preview and _preview_cancel():
                return buckets, scores, True
            frame = source.read()
            if frame is None:
                time.sleep(0.05)
                continue
            frame = mirror_bgr(frame)
            now_ms = int((time.monotonic() - origin) * 1000)
            if now_ms <= last_t:
                now_ms = last_t + 1
            last_t = now_ms
            hands = classifier.recognize(bgr_to_rgb(frame), now_ms)
            if hands:
                center = palm_center(hands[0])
                sample = PalmSample(float(center[0]), float(center[1]))
                if key == "rest":
                    buckets["rest"].append(sample)
                elif key in buckets:
                    buckets[key].append(sample)
                elif key.startswith("gesture:"):
                    gesture = key.split(":", 1)[1]
                    scores.setdefault(gesture, []).append(float(hands[0].score))
            if preview:
                remaining = max(0.0, deadline - time.monotonic())
                _draw_prompt(frame, prompt, remaining)
        if preview and _preview_cancel():
            return buckets, scores, True
    return buckets, scores, False


def _build_samples(buckets, scores):
    from gesture_media.calibration import CalibrationSamples, PalmSample

    def mean(points: list) -> PalmSample | None:
        if not points:
            return None
        return PalmSample(
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )

    rest = tuple(buckets["rest"])
    score_seq = {name: tuple(values) for name, values in scores.items()}
    score_mean = {
        name: (sum(values) / len(values) if values else 0.0) for name, values in scores.items()
    }
    attempts = [
        {
            "rest": rest,
            "top": mean(buckets["top"]),
            "bottom": mean(buckets["bottom"]),
            "left": mean(buckets["left"]),
            "right": mean(buckets["right"]),
            "gesture_scores": score_seq,
        },
        {
            "rest": rest,
            "top": tuple(buckets["top"]),
            "bottom": tuple(buckets["bottom"]),
            "left": tuple(buckets["left"]),
            "right": tuple(buckets["right"]),
            "gesture_scores": score_seq,
        },
        {
            "rest": rest,
            "top": tuple(buckets["top"]),
            "bottom": tuple(buckets["bottom"]),
            "left": tuple(buckets["left"]),
            "right": tuple(buckets["right"]),
            "gesture_scores": score_mean,
        },
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return CalibrationSamples(**kwargs)
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _zone_text(zone) -> str:
    if hasattr(zone, "x_min"):
        return (
            f"x {zone.x_min:.2f}-{zone.x_max:.2f}, y {zone.y_min:.2f}-{zone.y_max:.2f}"
        )
    return str(zone)


def _summarize_and_maybe_save(config, buckets, scores, *, preview: bool) -> bool:
    from gesture_media.calibration import compute_calibration
    from gesture_media.config import (
        PROFILE_VERSION,
        CalibrationProfile,
        Zone,
        profile_path,
        save_profile,
    )

    samples = _build_samples(buckets, scores)
    result = compute_calibration(samples, min_score=config.vision.min_score)
    print(f"deadzone: {config.engine.deadzone:.3f} -> {result.deadzone:.3f}")
    print(f"max_disp: {config.engine.max_disp:.3f} -> {result.max_disp:.3f}")
    print(f"zone: {_zone_text(config.vision.zone)} -> {_zone_text(result.zone)}")
    for warning in getattr(result, "warnings", ()):
        print(f"warning: {warning}")
    print("Save calibration? [y/n]")
    if preview:
        _draw_prompt_message("Save calibration? y / n / Esc")
    if not _confirm(preview):
        return False
    zone = result.zone
    if not hasattr(zone, "x_min"):
        x_min, x_max, y_min, y_max = zone
        zone = Zone(float(x_min), float(x_max), float(y_min), float(y_max))
    profile = CalibrationProfile(
        version=PROFILE_VERSION,
        deadzone=float(result.deadzone),
        max_disp=float(result.max_disp),
        zone=zone,
        min_score=None,
    )
    destination = profile_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_profile(destination, profile)
    print(f"Saved {destination}")
    return True


def _confirm(preview: bool) -> bool:
    if not preview:
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer == "y"
    import cv2

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("y"), ord("Y")):
            return True
        if key in (ord("n"), ord("N"), 27):
            return False
        line = _stdin_line()
        if line == "y":
            return True
        if line == "n":
            return False


def _stdin_line() -> str:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return ""
    if not readable:
        return ""
    try:
        return sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"


def _try_window() -> bool:
    try:
        import cv2

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        return True
    except Exception as exc:
        log.warning("calibration preview is unavailable (%s); prompts stay on the terminal", exc)
        return False


def _destroy_window() -> None:
    try:
        import cv2

        cv2.destroyWindow(WINDOW)
    except Exception:
        log.debug("calibration window close failed", exc_info=True)


def _draw_prompt(frame, text: str, remaining_s: float) -> None:
    import cv2

    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 96), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        text,
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{remaining_s:.1f}s   Esc cancels",
        (16, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(WINDOW, image)


def _draw_prompt_message(text: str) -> None:
    import cv2
    import numpy as np

    canvas = np.zeros((120, 900, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (16, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(WINDOW, canvas)


def _preview_cancel() -> bool:
    import cv2

    key = cv2.waitKey(1) & 0xFF
    return key == 27


def _notify(config, detail: str) -> None:
    try:
        from gesture_media.config import should_notify
        from gesture_media.notify import DesktopNotifier
    except ImportError:
        log.info("calibration notification skipped; config is not importable")
        return
    if not should_notify(config, "calibration"):
        return
    notifier = DesktopNotifier()
    try:
        body = "Calibration cancelled" if detail == "cancelled" else "Calibration saved"
        notifier.notify("Gesture Media Controller", body, urgency="low", expire_ms=3000)
    finally:
        notifier.close()
