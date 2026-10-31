"""Run loop, replay, camera listing, and the single-instance lock.

Capture, inference, and the preview stay on the main thread. Player commands
go through ``ActionDispatcher`` on its worker. OpenCV calls stay on this thread.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TextIO

from gesture_media.action_types import Action, Kind
from gesture_media.players.base import Capability

log = logging.getLogger(__name__)

WINDOW = "Gesture Media Controller"
_KIND_CAPABILITY = {
    Kind.PLAY_PAUSE: Capability.PLAY_PAUSE,
    Kind.NEXT: Capability.NEXT,
    Kind.PREVIOUS: Capability.PREVIOUS,
    Kind.SEEK: Capability.SEEK,
    Kind.VOLUME: Capability.VOLUME,
    Kind.MUTE_TOGGLE: Capability.MUTE,
    Kind.PITCH: Capability.PITCH,
    Kind.PITCH_RESET: Capability.PITCH,
    Kind.OPEN_PLAYER: Capability.OPEN,
}
_KIND_EVENT = {
    Kind.PLAY_PAUSE: "play_pause",
    Kind.NEXT: "next",
    Kind.PREVIOUS: "previous",
    Kind.MUTE_TOGGLE: "mute",
    Kind.PITCH_RESET: "pitch_reset",
    Kind.OPEN_PLAYER: "open_player",
}


@dataclass
class InstanceLock:
    handle: TextIO
    path: Path

    def release(self) -> None:
        import fcntl

        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.handle.close()
        try:
            self.path.unlink()
        except OSError:
            pass


def acquire_instance_lock(directory: Path) -> InstanceLock | None:
    """Exclusive lock under ``directory``. A second caller gets None."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gesture-media.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        print("gesture-media is already running.", file=sys.stderr)
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return InstanceLock(handle, path)


def describe_action(action: Action, *, volume_target: str) -> str:
    gesture = action.gesture
    if action.kind == Kind.VOLUME:
        return f"volume {_signed(action.value)}% ({volume_target}) {gesture}"
    if action.kind == Kind.SEEK:
        return f"seek {_signed(action.value)} s {gesture}"
    if action.kind == Kind.PITCH:
        return f"pitch {_signed(action.value)} semitones {gesture}"
    if action.kind == Kind.PLAY_PAUSE:
        return f"play/pause {gesture}"
    if action.kind == Kind.MUTE_TOGGLE:
        return f"mute {gesture}"
    if action.kind == Kind.NEXT:
        return f"next {gesture}"
    if action.kind == Kind.PREVIOUS:
        return f"previous {gesture}"
    if action.kind == Kind.PITCH_RESET:
        return f"pitch reset {gesture}"
    if action.kind == Kind.OPEN_PLAYER:
        return f"open player {gesture}"
    return f"{action.kind.value} {gesture}"


def format_action_log(action: Action, *, volume_target: str, stamp: str | None = None) -> str:
    when = stamp if stamp is not None else time.strftime("%H:%M:%S")
    return f"{when} ACTION {describe_action(action, volume_target=volume_target)}"


def _signed(value: float | None) -> str:
    number = 0.0 if value is None else float(value)
    rounded = round(number, 1)
    if float(rounded).is_integer():
        rendered = str(int(rounded))
    else:
        rendered = f"{rounded:.1f}"
    if rounded > 0:
        return f"+{rendered}"
    return rendered


@dataclass
class _Control:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: threading.Event = field(default_factory=threading.Event)
    toggle_pause: bool = False
    toggle_preview: bool = False
    calibrate: bool = False
    mode: str | None = None
    toggle_live: bool = False
    toggle_final: bool = False


def _zone_tuple(zone) -> tuple[float, float, float, float]:
    return (float(zone.x_min), float(zone.y_min), float(zone.x_max), float(zone.y_max))


def _audio_like(obj: object) -> bool:
    names = ("volume_delta", "toggle_mute", "get_state")
    return all(callable(getattr(obj, name, None)) for name in names)


class _Session:
    """Mutable pieces of one run. Kept together so the loop stays readable."""

    def __init__(self, config, config_path: Path | None, record_path: Path | None) -> None:
        self.config = config
        self.config_path = config_path
        self.record_path = record_path
        self.control = _Control()
        self.notices: list[str] = []
        self.notice_lock = threading.Lock()
        self.pending: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.unsupported: set[tuple[str, str]] = set()
        self.baseline_lock = threading.Lock()
        self.baseline_volume: float | None = None
        self.baseline_muted: bool | None = None
        self.engagement_volume: float | None = None
        self.engagement_kind: str | None = None
        self.hud_enabled = bool(config.hud.enabled)
        self.show_landmarks = bool(config.hud.show_landmarks)
        self.preview = not bool(config.no_preview)
        self.window_open = False
        self.last_action: str | None = None
        self.fps = 0.0
        self.exit_code = 0
        self.started = False
        self.camera_lost = False
        self.camera_held = False
        self.backend_lost = False
        self.paused_known = False
        self.backend_was_up: bool | None = None
        self.record_file: IO[str] | None = None
        self.notice_state = None
        self.notifier = None
        self.player = None
        self.audio = None
        self.backend_name = ""
        self.volume_target = "system"
        self.dispatcher = None
        self.engine = None
        self.classifier = None
        self.source = None
        self.tray = None
        self.watcher_stop = threading.Event()
        self.watcher: threading.Thread | None = None

    def add_notice(self, text: str) -> None:
        with self.notice_lock:
            if text not in self.notices:
                self.notices.append(text)
                del self.notices[:-8]

    def notice_snapshot(self) -> tuple[str, ...]:
        with self.notice_lock:
            return tuple(self.notices)

    def remember_audio(self, volume: float, muted: bool) -> None:
        with self.baseline_lock:
            self.baseline_volume = volume
            self.baseline_muted = muted

    def audio_snapshot(self) -> tuple[float | None, bool | None]:
        with self.baseline_lock:
            return self.baseline_volume, self.baseline_muted


def _emit(session: _Session, event: str, **kwargs) -> None:
    from gesture_media.notify import apply_notices, notification_allowed, plan_notifications

    if session.notice_state is None or session.notifier is None:
        return
    try:
        from gesture_media.config import should_notify

        allowed = should_notify(session.config, event)
    except ImportError:
        allowed = notification_allowed(session.config, event)
    if not allowed:
        return
    session.notice_state, planned = plan_notifications(
        session.config,
        session.notice_state,
        event,
        prefiltered=True,
        **kwargs,
    )
    session.notice_state = apply_notices(session.notifier, session.notice_state, planned)


def _drain_pending(session: _Session, t_ms: int) -> None:
    while True:
        try:
            event, payload = session.pending.get_nowait()
        except queue.Empty:
            return
        payload.setdefault("t_ms", t_ms)
        _emit(session, event, **payload)


def _note_unsupported(session: _Session, backend: str, capability: str) -> None:
    key = (backend, capability)
    if key in session.unsupported:
        return
    session.unsupported.add(key)
    log.warning("%s is not supported on %s", capability, backend)
    session.add_notice(f"{capability} unsupported on {backend}")
    if capability == "pitch":
        session.add_notice("pitch unsupported on this backend")
    session.pending.put(("unsupported", {"capability": capability, "backend": backend}))


def _player_for(session: _Session, capability: Capability):
    player = session.player
    if hasattr(player, "routed_player"):
        return player.routed_player(capability)
    return player


def _audio_for(session: _Session, capability: Capability):
    player = session.player
    routing = getattr(player, "routing", {})
    routed_name = routing.get(capability.value)
    if routed_name:
        candidate = getattr(player, "players", {}).get(routed_name)
        if _audio_like(candidate):
            return candidate
        _note_unsupported(session, str(routed_name), capability.value)
        return None
    return session.audio


def _refresh_audio(session: _Session, audio) -> None:
    if audio is None:
        return
    try:
        volume, muted = audio.get_state()
    except Exception:
        log.debug("audio state unavailable", exc_info=True)
        return
    session.remember_audio(float(volume), bool(muted))


def _perform(session: _Session, action: Action) -> None:
    capability = _KIND_CAPABILITY[action.kind]
    if action.kind in {Kind.VOLUME, Kind.MUTE_TOGGLE}:
        audio = _audio_for(session, capability)
        if audio is None:
            return
        if action.kind == Kind.VOLUME:
            audio.volume_delta(float(action.value or 0.0))
        else:
            audio.toggle_mute()
        _refresh_audio(session, audio)
        return
    player = _player_for(session, capability)
    backend = getattr(player, "name", session.backend_name)
    if capability not in getattr(player, "capabilities", frozenset()):
        _note_unsupported(session, backend, capability.value)
        return
    if action.kind == Kind.PLAY_PAUSE:
        player.play_pause()
    elif action.kind == Kind.NEXT:
        player.next()
    elif action.kind == Kind.PREVIOUS:
        player.previous()
    elif action.kind == Kind.SEEK:
        player.seek(float(action.value or 0.0))
    elif action.kind == Kind.PITCH:
        player.set_pitch(float(action.value or 0.0))
    elif action.kind == Kind.PITCH_RESET:
        player.set_pitch(0.0)
    elif action.kind == Kind.OPEN_PLAYER:
        player.open()


def _notify_discrete(session: _Session, action: Action, t_ms: int) -> None:
    event = _KIND_EVENT.get(action.kind)
    if event is None:
        return
    muted = None
    if action.kind == Kind.MUTE_TOGGLE:
        _volume, current = session.audio_snapshot()
        muted = None if current is None else not current
        if muted is not None:
            with session.baseline_lock:
                session.baseline_muted = muted
    _emit(
        session,
        event,
        t_ms=t_ms,
        gesture=action.gesture,
        muted=muted,
        backend=session.backend_name,
    )


def _notify_engagement(session: _Session, engagement, t_ms: int) -> None:
    kind = engagement.kind.value
    if session.engagement_kind != kind or engagement.ended:
        if session.engagement_kind != kind:
            volume, _muted = session.audio_snapshot()
            session.engagement_volume = volume
        session.engagement_kind = None if engagement.ended else kind
    value: float | None = None
    position = None
    if engagement.kind == Kind.VOLUME:
        base = 0.0 if session.engagement_volume is None else session.engagement_volume
        value = min(100.0, max(0.0, base + float(engagement.accumulated)))
    elif engagement.kind == Kind.SEEK:
        value = float(engagement.accumulated)
        if engagement.ended and session.player is not None:
            try:
                position = session.player.get_position()
            except Exception:
                log.debug("position unavailable", exc_info=True)
                position = None
    elif engagement.kind == Kind.PITCH:
        value = (
            float(engagement.absolute)
            if engagement.absolute is not None
            else float(session.engine.pitch_semitones)
        )
    _emit(
        session,
        kind,
        t_ms=t_ms,
        gesture=engagement.gesture,
        value=value,
        changed=bool(engagement.changed),
        ended=bool(engagement.ended),
        position_s=position,
        backend=session.backend_name,
    )


def _replace_notifications(session: _Session, **changes) -> None:
    from dataclasses import replace

    session.config = replace(
        session.config,
        notifications=replace(session.config.notifications, **changes),
    )


def _apply_control(session: _Session) -> None:
    with session.control.lock:
        toggle_pause = session.control.toggle_pause
        toggle_preview = session.control.toggle_preview
        calibrate = session.control.calibrate
        mode = session.control.mode
        toggle_live = session.control.toggle_live
        toggle_final = session.control.toggle_final
        session.control.toggle_pause = False
        session.control.toggle_preview = False
        session.control.calibrate = False
        session.control.mode = None
        session.control.toggle_live = False
        session.control.toggle_final = False
    if mode is not None:
        _replace_notifications(session, verbosity=mode)
    if toggle_live:
        _replace_notifications(session, live_updates=not session.config.notifications.live_updates)
    if toggle_final:
        _replace_notifications(session, final_value=not session.config.notifications.final_value)
    if toggle_preview:
        session.preview = not session.preview
    if toggle_pause and session.engine is not None:
        session.engine.set_paused(not session.engine.paused)
        _mark_paused(session, session.engine.paused, "")
    if calibrate:
        _run_calibration_from_tray(session)
    _sync_tray(session)


def _mark_paused(session: _Session, paused: bool, gesture: str) -> None:
    if session.paused_known == paused:
        _sync_tray(session)
        return
    session.paused_known = paused
    _emit(session, "pause" if paused else "resume", gesture=gesture)
    _sync_tray(session)


def _sync_tray(session: _Session) -> None:
    if session.tray is None:
        return
    if session.camera_lost or session.backend_lost:
        state = "error"
    elif session.engine is not None and session.engine.paused:
        state = "paused"
    else:
        state = "active"
    session.tray.set_state(state)


def _release_source(session: _Session) -> None:
    session.camera_held = False
    source = session.source
    session.source = None
    if source is None:
        return
    try:
        source.release()
    except Exception:
        log.debug("camera release failed", exc_info=True)


def _open_source(session: _Session, open_source, camera_open_error) -> bool:
    try:
        session.source = open_source(session.config)
    except camera_open_error as exc:
        print(f"Could not open the camera: {exc}", file=sys.stderr)
        print("Check `gesture-media list-cameras` and camera.index.", file=sys.stderr)
        session.exit_code = 2
        session.camera_held = False
        return False
    session.camera_held = True
    return True


def _close_window(session: _Session) -> None:
    if not session.window_open:
        return
    import cv2

    session.window_open = False
    try:
        cv2.destroyWindow(WINDOW)
    except Exception:
        log.debug("destroyWindow failed", exc_info=True)


def _run_calibration_from_tray(session: _Session) -> None:
    from gesture_media.calibrate_ui import run_calibration

    _close_window(session)
    _release_source(session)
    if session.classifier is not None:
        try:
            session.classifier.close()
        except Exception:
            log.debug("classifier close before calibrate failed", exc_info=True)
        session.classifier = None
    run_calibration(session.config, camera=session.config.camera.index)
    _open_classifier(session)


def _open_classifier(session: _Session) -> None:
    if session.config.vision.classifier == "rules":
        from gesture_media.vision.rules import RulesClassifier

        session.classifier = RulesClassifier()
        return
    from gesture_media.vision.classifier import MediaPipeClassifier

    session.classifier = MediaPipeClassifier(
        model_path(),
        num_hands=2,
        min_score=session.config.vision.min_score,
    )


def model_path() -> Path:
    cwd_candidate = Path.cwd() / "models" / "gesture_recognizer.task"
    if cwd_candidate.is_file():
        return cwd_candidate
    return Path(__file__).resolve().parents[2] / "models" / "gesture_recognizer.task"


def _hooks(session: _Session):
    from gesture_media.tray import TrayHooks

    control = session.control

    def toggle_pause() -> None:
        with control.lock:
            control.toggle_pause = True

    def toggle_preview() -> None:
        with control.lock:
            control.toggle_preview = True

    def calibrate() -> None:
        with control.lock:
            control.calibrate = True

    def open_config() -> None:
        import subprocess

        path = session.config_path or Path.cwd() / "config.yaml"
        try:
            subprocess.Popen(
                ["xdg-open", str(path)],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.warning("could not open %s (%s)", path, exc)

    def set_mode(mode: str) -> None:
        with control.lock:
            control.mode = mode

    def toggle_live() -> None:
        with control.lock:
            control.toggle_live = True

    def toggle_final() -> None:
        with control.lock:
            control.toggle_final = True

    def quit_app() -> None:
        control.stop.set()

    return TrayHooks(
        toggle_pause=toggle_pause,
        toggle_preview=toggle_preview,
        calibrate=calibrate,
        open_config=open_config,
        set_notifications=set_mode,
        toggle_live_updates=toggle_live,
        toggle_final_value=toggle_final,
        quit=quit_app,
        paused=lambda: bool(session.engine and session.engine.paused),
        preview_visible=lambda: session.preview,
        notification_mode=lambda: session.config.notifications.verbosity,
        live_updates=lambda: bool(session.config.notifications.live_updates),
        final_value=lambda: bool(session.config.notifications.final_value),
    )


def _handle_key(session: _Session, key: int) -> None:
    if key in (ord("q"), 27):
        session.control.stop.set()
    elif key == ord("d") and session.engine is not None:
        enabled = session.engine.set_detection_enabled(not session.engine.detection_enabled)
        log.info("detection %s", "on" if enabled else "off")
    elif key == ord("h"):
        session.hud_enabled = not session.hud_enabled
    elif key == ord("l"):
        session.show_landmarks = not session.show_landmarks


def _show_frame(session: _Session, frame, result, hands) -> None:
    import cv2

    from gesture_media.hud import draw_hud
    from gesture_media.hud_model import HudModel

    if not session.window_open:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        session.window_open = True
    volume, muted = session.audio_snapshot()
    model = HudModel(
        gesture=result.active_gesture,
        score=result.active_score,
        hold_progress=result.hold_progress,
        last_action=session.last_action,
        fps=session.fps,
        volume=volume,
        muted=muted,
        pitch_semitones=result.pitch_semitones,
        backend=session.backend_name,
        notices=session.notice_snapshot(),
        zone=_zone_tuple(session.config.vision.zone),
        show_landmarks=session.show_landmarks,
        hands=hands,
        calibrated=bool(session.config.calibration_active),
        paused=result.paused,
        detection_enabled=result.detection_enabled,
        vinyl_mode=bool(getattr(session.player, "vinyl_mode", False)),
        hud_enabled=session.hud_enabled,
    )
    drawn = draw_hud(frame, model) if session.hud_enabled else frame
    cv2.imshow(WINDOW, drawn)
    _handle_key(session, cv2.waitKey(1) & 0xFF)


def _watch_backend(session: _Session) -> None:
    while not session.watcher_stop.is_set():
        player = session.player
        if player is not None:
            try:
                available = bool(player.is_available())
            except Exception:
                available = False
            was = session.backend_was_up
            session.backend_was_up = available
            session.backend_lost = not available
            if was is not False and not available:
                session.pending.put(
                    ("error", {"detail": "Player backend lost", "backend": session.backend_name})
                )
                session.add_notice("player backend lost")
        if session.watcher_stop.wait(5.0):
            return


def _install_signals(stop: threading.Event) -> None:
    import signal

    def _handle(signum, _frame) -> None:
        log.info("received signal %s", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def run(config, *, config_path: Path | None = None, record_path: Path | None = None) -> int:
    """Capture, recognize, and dispatch. Exit 0, or 2 if the camera fails."""
    try:
        from gesture_media.actions import ActionDispatcher
        from gesture_media.config import resolve_volume_target
        from gesture_media.engine import GestureEngine
        from gesture_media.sources import CameraOpenError, bgr_to_rgb, mirror_bgr, open_source
    except ImportError as exc:
        print(
            "gesture-media run needs the engine, config, actions, and sources modules "
            f"({exc}).",
            file=sys.stderr,
        )
        return 1
    from gesture_media.notify import DesktopNotifier, NoticeState
    from gesture_media.players.registry import build_backends
    from gesture_media.tray import start_tray
    from gesture_media.vision.types import frame_to_json

    session = _Session(config, config_path, record_path)
    _install_signals(session.control.stop)
    if not _open_source(session, open_source, CameraOpenError):
        return session.exit_code
    try:
        if session.config.vision.classifier == "mediapipe" and not model_path().is_file():
            print(f"Model file is missing: {model_path()}", file=sys.stderr)
            print("Run scripts/fetch_model.py and try again.", file=sys.stderr)
            session.exit_code = 1
            return session.exit_code
        _open_classifier(session)
        session.engine = GestureEngine(config)
        session.player, session.audio, session.backend_name = build_backends(config)
        session.volume_target = resolve_volume_target(config, session.backend_name)
        session.notifier = DesktopNotifier()
        session.notice_state = NoticeState()
        session.dispatcher = ActionDispatcher(lambda action: _perform(session, action))
        _refresh_audio(session, session.audio)
        session.started = True
        _emit(session, "app_on", backend=session.backend_name)
        if config.tray.enabled:
            session.tray = start_tray(_hooks(session), backend_name=session.backend_name)
        session.watcher = threading.Thread(
            target=_watch_backend, args=(session,), name="backend-watch", daemon=True
        )
        session.watcher.start()
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            session.record_file = record_path.open("w", encoding="utf-8")
        _loop(session, bgr_to_rgb, mirror_bgr, frame_to_json)
    except Exception:
        log.exception("run failed")
        session.exit_code = 1
    finally:
        _shutdown(session)
    return session.exit_code


def _loop(session: _Session, bgr_to_rgb, mirror_bgr, frame_to_json) -> None:
    from gesture_media.sources import CameraOpenError, open_source

    started = time.monotonic()
    last_t = -1
    fps_mark = started
    fps_frames = 0
    while not session.control.stop.is_set():
        _apply_control(session)
        if session.control.stop.is_set():
            break
        paused = bool(session.engine.paused)
        release_when_paused = bool(session.config.tray.release_camera_when_paused)
        if paused and release_when_paused:
            if session.camera_held:
                _release_source(session)
                log.info("camera released while detection is paused")
            session.control.stop.wait(0.05)
            continue
        if not session.camera_held:
            if not _open_source(session, open_source, CameraOpenError):
                break
        frame_started = time.monotonic()
        frame = session.source.read()
        if frame is None:
            if session.config.source:
                log.info("source ended")
                break
            if not session.camera_lost:
                session.camera_lost = True
                _emit(session, "error", detail="Camera lost", backend=session.backend_name)
                session.add_notice("camera lost")
                _sync_tray(session)
            session.control.stop.wait(0.2)
            continue
        if session.camera_lost:
            session.camera_lost = False
            _sync_tray(session)
        frame = mirror_bgr(frame)
        t_ms = int((time.monotonic() - started) * 1000)
        if t_ms <= last_t:
            t_ms = last_t + 1
        last_t = t_ms
        hands = session.classifier.recognize(bgr_to_rgb(frame), t_ms)
        if session.record_file is not None:
            session.record_file.write(frame_to_json(t_ms, hands) + "\n")
            session.record_file.flush()
        result = session.engine.step(hands, t_ms)
        if session.config.verbose:
            log.debug(
                "frame t_ms=%s hands=%s gesture=%s score=%s",
                t_ms,
                len(hands),
                result.active_gesture,
                result.active_score,
            )
        if result.paused != session.paused_known:
            gesture = session.config.pause.gesture if result.pause_toggled else ""
            _mark_paused(session, result.paused, gesture)
        for action in result.actions:
            line = format_action_log(action, volume_target=session.volume_target)
            print(line, flush=True)
            session.last_action = describe_action(action, volume_target=session.volume_target)
            _notify_discrete(session, action, t_ms)
            session.dispatcher.submit(action)
        if result.engagement is not None:
            _notify_engagement(session, result.engagement, t_ms)
        _drain_pending(session, t_ms)
        fps_frames += 1
        now = time.monotonic()
        if now - fps_mark >= 0.5:
            session.fps = fps_frames / (now - fps_mark)
            fps_frames = 0
            fps_mark = now
        if session.preview:
            try:
                _show_frame(session, frame, result, hands)
            except Exception:
                log.exception("preview failed; continuing headless. Try QT_QPA_PLATFORM=xcb")
                session.preview = False
                _close_window(session)
        elif session.window_open:
            _close_window(session)
        _sync_tray(session)
        target_fps = session.config.pause.paused_fps if result.paused else session.config.camera.fps
        budget = 1.0 / max(float(target_fps), 1.0)
        remaining = budget - (time.monotonic() - frame_started)
        if remaining > 0:
            session.control.stop.wait(remaining)


def _shutdown(session: _Session) -> None:
    if session.started:
        try:
            _emit(session, "app_off", backend=session.backend_name)
        except Exception:
            log.debug("stop notification failed", exc_info=True)
    _release_source(session)
    if session.player is not None:
        try:
            session.player.close()
        except Exception:
            log.debug("player close failed", exc_info=True)
    if session.dispatcher is not None:
        try:
            session.dispatcher.shutdown()
        except Exception:
            log.debug("dispatcher shutdown failed", exc_info=True)
    if session.classifier is not None:
        try:
            session.classifier.close()
        except Exception:
            log.debug("classifier close failed", exc_info=True)
    session.watcher_stop.set()
    if session.watcher is not None:
        session.watcher.join(timeout=2.0)
    _close_window(session)
    if session.window_open or session.preview:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            log.debug("destroyAllWindows failed", exc_info=True)
    if session.tray is not None:
        session.tray.close()
    if session.notifier is not None:
        session.notifier.close()
    if session.record_file is not None:
        session.record_file.close()


def list_cameras() -> int:
    """Probe indexes 0 through 8, print the ones that open, and release each."""
    try:
        import cv2
    except ImportError as exc:
        print(f"OpenCV is not importable ({exc}).", file=sys.stderr)
        return 1
    found = False
    for index in range(9):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                print(index)
                found = True
        finally:
            capture.release()
    if not found:
        print("No cameras opened")
    return 0


def replay(path: Path, config=None) -> int:
    """Feed a JSONL recording through ``GestureEngine`` and a ``FakePlayer``."""
    try:
        from gesture_media.config import load_config
        from gesture_media.engine import GestureEngine
        from gesture_media.players.fake import FakeAudio, FakePlayer
        from gesture_media.vision.types import frame_from_json
    except ImportError as exc:
        print(f"replay needs the engine, config, and fake player ({exc}).", file=sys.stderr)
        return 1
    if config is None:
        config = load_config(use_profile=False)
    engine = GestureEngine(config)
    player = FakePlayer()
    audio = FakeAudio()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return 1
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            t_ms, hands = frame_from_json(line)
            result = engine.step(hands, t_ms)
        except ValueError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 1
        for action in result.actions:
            _drive_fake(player, audio, action)
            print(format_action_log(action, volume_target="player"), flush=True)
    return 0


def _drive_fake(player, audio, action: Action) -> None:
    if action.kind == Kind.PLAY_PAUSE:
        player.play_pause()
    elif action.kind == Kind.NEXT:
        player.next()
    elif action.kind == Kind.PREVIOUS:
        player.previous()
    elif action.kind == Kind.SEEK:
        player.seek(action.value)
    elif action.kind == Kind.VOLUME:
        audio.volume_delta(action.value)
    elif action.kind == Kind.MUTE_TOGGLE:
        audio.toggle_mute()
    elif action.kind == Kind.PITCH:
        player.set_pitch(action.value)
    elif action.kind == Kind.PITCH_RESET:
        player.set_pitch(0)
    elif action.kind == Kind.OPEN_PLAYER:
        player.open()
