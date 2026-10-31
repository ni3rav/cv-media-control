"""Desktop notifications and the section 20/21 delivery policy.

``plan_notifications`` is pure. ``DesktopNotifier`` sends one notice and never
raises. The app calls ``gesture_media.config.should_notify`` and then plans
with ``prefiltered=True``. Tests pass a duck-typed config and use
``notification_allowed``, which mirrors that policy, so they do not import the
config module.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_SEND_TIMEOUT_S = 0.5
_GESTURE_LABELS = {
    "Open_Palm": "Open palm",
    "Closed_Fist": "Fist",
    "Thumb_Up": "Thumb up",
    "Thumb_Down": "Thumb down",
    "Thumb_Right": "Thumb right",
    "Thumb_Left": "Thumb left",
    "Pointing_Up": "Pointing up",
    "Victory": "Peace sign",
    "ILoveYou": "I love you",
    "swipe_right": "Swipe right",
    "swipe_left": "Swipe left",
    "two_fists": "two fists",
}
_IMPORTANT = frozenset({"app_on", "app_off", "pause", "resume", "error"})
_CONTINUOUS = frozenset({"volume", "seek", "pitch"})


def gesture_label(name: str | None) -> str:
    if not name:
        return ""
    return _GESTURE_LABELS.get(name, name.replace("_", " "))


def urgency_byte(urgency: str) -> int:
    """Freedesktop urgency hint. Low is 0, normal is 1."""
    return 1 if urgency == "normal" else 0


def notification_allowed(config, event: str) -> bool:
    """Same rules as ``config.should_notify`` for a duck-typed config."""
    notes = config.notifications
    if not getattr(notes, "enabled", True) or getattr(notes, "verbosity", "all") == "off":
        return False
    verbosity = notes.verbosity
    if verbosity == "important" and event not in _IMPORTANT:
        return False
    if event == "app_on" and not getattr(notes, "on_start", True):
        return False
    if event in {"pause", "resume"} and not getattr(notes, "on_toggle", True):
        return False
    if event == "error" and not getattr(notes, "on_error", True):
        return False
    per_action = getattr(notes, "per_action", None) or {}
    if verbosity == "all" and event in per_action and not per_action[event]:
        return False
    return True


def _fmt_number(value: float, *, signed: bool) -> str:
    rounded = round(float(value), 1)
    if float(rounded).is_integer():
        rendered = str(int(rounded))
    else:
        rendered = f"{rounded:.1f}"
    if signed and rounded > 0:
        return f"+{rendered}"
    return rendered


def _fmt_position(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    return f"{whole // 60}:{whole % 60:02d}"


@dataclass(frozen=True)
class PlannedNotice:
    summary: str
    body: str
    urgency: str
    expire_ms: int
    slot: str | None
    event: str


@dataclass
class NoticeState:
    last_ms: dict[str, int] = field(default_factory=dict)
    unsupported: set[tuple[str, str]] = field(default_factory=set)
    slots: dict[str, int] = field(default_factory=dict)

    def copy(self) -> NoticeState:
        return NoticeState(dict(self.last_ms), set(self.unsupported), dict(self.slots))


def _summary() -> str:
    return "Gesture Media Controller"


def _discrete_body(
    event: str,
    *,
    backend: str,
    gesture: str,
    muted: bool | None,
    detail: str,
    capability: str,
) -> str:
    label = gesture_label(gesture)
    if event == "app_on":
        return f"Gesture Media Controller started. Active, backend: {backend}"
    if event == "app_off":
        return "Gesture Media Controller stopped"
    if event == "pause":
        return f"Detection paused ({label})" if label else "Detection paused"
    if event == "resume":
        return f"Detection resumed ({label})" if label else "Detection resumed"
    if event == "play_pause":
        return f"Play/Pause ({label})" if label else "Play/Pause"
    if event == "mute":
        text = f"Mute toggled ({label})" if label else "Mute toggled"
        if muted is True:
            return f"{text}. Muted"
        if muted is False:
            return f"{text}. Unmuted"
        return text
    if event == "next":
        return f"Next track ({label})" if label else "Next track"
    if event == "previous":
        return f"Previous track ({label})" if label else "Previous track"
    if event == "pitch_reset":
        return "Pitch reset"
    if event == "open_player":
        return "Opening player"
    if event == "unsupported":
        pretty = capability.replace("_", " ").capitalize() if capability else "Action"
        return f"{pretty} not supported on {backend}"
    if event == "error":
        return detail or "Something went wrong"
    if event == "calibration":
        if detail == "cancelled":
            return "Calibration cancelled"
        return "Calibration saved"
    return detail or event


def _live_body(event: str, value: float, gesture: str) -> str:
    label = gesture_label(gesture)
    if event == "volume":
        return f"Volume {_fmt_number(value, signed=False)}% ({label})"
    if event == "seek":
        return f"Seek {_fmt_number(value, signed=True)} s ({label})"
    return f"Pitch {_fmt_number(value, signed=True)} semitones ({label})"


def _final_body(event: str, value: float, position_s: float | None) -> str:
    if event == "volume":
        return f"Volume set to {_fmt_number(value, signed=False)}%"
    if event == "seek":
        text = f"Seek total {_fmt_number(value, signed=True)} s"
        if position_s is not None:
            return f"{text}, now at {_fmt_position(position_s)}"
        return text
    return f"Pitch set to {_fmt_number(value, signed=True)} semitones"


def plan_notifications(
    config,
    state: NoticeState,
    event: str,
    *,
    t_ms: int = 0,
    backend: str = "",
    gesture: str = "",
    value: float | None = None,
    changed: bool = False,
    ended: bool = False,
    position_s: float | None = None,
    muted: bool | None = None,
    capability: str = "",
    detail: str = "",
    prefiltered: bool = False,
) -> tuple[NoticeState, tuple[PlannedNotice, ...]]:
    """Decide the notices for one event. ``state`` is not mutated."""
    notes = config.notifications
    if not prefiltered and not notification_allowed(config, event):
        return state.copy(), ()
    new = state.copy()
    if event == "unsupported":
        key = (capability, backend)
        if key in new.unsupported:
            return new, ()
        new.unsupported.add(key)
    urgency = "normal" if event == "error" else "low"
    discrete_ms = int(getattr(notes, "expire_discrete_ms", 3000))
    continuous_ms = int(getattr(notes, "expire_continuous_ms", 1500))
    if event in _CONTINUOUS:
        amount = 0.0 if value is None else float(value)
        slot = event
        if ended:
            new.last_ms.pop(slot, None)
            if not getattr(notes, "final_value", True) or not changed:
                return new, ()
            notice = PlannedNotice(
                summary=_summary(),
                body=_final_body(event, amount, position_s),
                urgency=urgency,
                expire_ms=discrete_ms,
                slot=slot,
                event=event,
            )
            return new, (notice,)
        if not getattr(notes, "live_updates", True):
            return new, ()
        update_hz = float(getattr(notes, "update_hz", 5) or 5)
        interval = max(1, int(round(1000 / update_hz)))
        last = new.last_ms.get(slot)
        if last is not None and t_ms - last < interval:
            return new, ()
        new.last_ms[slot] = t_ms
        notice = PlannedNotice(
            summary=_summary(),
            body=_live_body(event, amount, gesture),
            urgency=urgency,
            expire_ms=continuous_ms,
            slot=slot,
            event=event,
        )
        return new, (notice,)
    notice = PlannedNotice(
        summary=_summary(),
        body=_discrete_body(
            event,
            backend=backend,
            gesture=gesture,
            muted=muted,
            detail=detail,
            capability=capability,
        ),
        urgency=urgency,
        expire_ms=discrete_ms,
        slot=None,
        event=event,
    )
    return new, (notice,)


class Notifier:
    def notify(
        self,
        summary: str,
        body: str,
        *,
        urgency: str = "low",
        expire_ms: int = 3000,
        replace_id: int | None = None,
    ) -> int:
        raise NotImplementedError

    def close(self) -> None:
        return None


class FakeNotifier(Notifier):
    """Records calls, including the id that was replaced."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._next = 1
        self.closed = False

    def notify(
        self,
        summary: str,
        body: str,
        *,
        urgency: str = "low",
        expire_ms: int = 3000,
        replace_id: int | None = None,
    ) -> int:
        if replace_id:
            notice_id = replace_id
        else:
            notice_id = self._next
            self._next += 1
        self.calls.append(
            {
                "summary": summary,
                "body": body,
                "urgency": urgency,
                "expire_ms": expire_ms,
                "replace_id": replace_id,
                "id": notice_id,
            }
        )
        return notice_id

    def close(self) -> None:
        self.closed = True


def apply_notices(
    notifier: Notifier,
    state: NoticeState,
    planned: tuple[PlannedNotice, ...],
) -> NoticeState:
    """Send ``planned`` and remember replace ids. The notifier must not raise."""
    new = state.copy()
    for notice in planned:
        replace = new.slots.get(notice.slot) if notice.slot else None
        notice_id = notifier.notify(
            notice.summary,
            notice.body,
            urgency=notice.urgency,
            expire_ms=notice.expire_ms,
            replace_id=replace,
        )
        if notice.slot and notice_id:
            new.slots[notice.slot] = notice_id
    return new


def _notify_message(summary: str, body: str, urgency: str, expire_ms: int, replace_id: int | None):
    from jeepney import DBusAddress, new_method_call

    address = DBusAddress(
        "/org/freedesktop/Notifications",
        bus_name="org.freedesktop.Notifications",
        interface="org.freedesktop.Notifications",
    )
    hints = [
        ("urgency", ("y", urgency_byte(urgency))),
        ("suppress-sound", ("b", True)),
    ]
    return new_method_call(
        address,
        "Notify",
        "susssasa{sv}i",
        (
            "gesture-media",
            int(replace_id or 0),
            "",
            summary,
            body,
            [],
            hints,
            int(expire_ms),
        ),
    )


class DesktopNotifier(Notifier):
    """D-Bus Notify, then notify-send. A dead service is logged once and skipped."""

    def __init__(self) -> None:
        self._mode = "try"
        self._connection = None
        self._gave_up = False

    def _notify_dbus(
        self,
        *,
        summary: str,
        body: str,
        urgency: str,
        expire_ms: int,
        replace_id: int | None,
    ) -> int:
        from jeepney.io.blocking import open_dbus_connection

        if self._connection is None:
            self._connection = open_dbus_connection(bus="SESSION", auth_timeout=_SEND_TIMEOUT_S)
        message = _notify_message(summary, body, urgency, expire_ms, replace_id)
        reply = self._connection.send_and_get_reply(message, timeout=_SEND_TIMEOUT_S)
        return int(reply.body[0])

    def _notify_send(
        self,
        *,
        summary: str,
        body: str,
        urgency: str,
        expire_ms: int,
        replace_id: int | None,
    ) -> int:
        argv = [
            "notify-send",
            "--print-id",
            "--app-name=gesture-media",
            f"--urgency={urgency}",
            f"--expire-time={expire_ms}",
            "--hint=boolean:suppress-sound:true",
        ]
        if replace_id:
            argv.extend(["--replace-id", str(replace_id)])
        argv.extend([summary, body])
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=_SEND_TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"notify-send exited {proc.returncode}")
        token = proc.stdout.strip().split()
        if not token:
            raise RuntimeError("notify-send did not print an id")
        return int(token[0])

    def _drop_connection(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            log.debug("closing the notification bus failed", exc_info=True)

    def notify(
        self,
        summary: str,
        body: str,
        *,
        urgency: str = "low",
        expire_ms: int = 3000,
        replace_id: int | None = None,
    ) -> int:
        if self._mode == "off":
            return 0
        kwargs = {
            "summary": summary,
            "body": body,
            "urgency": urgency,
            "expire_ms": expire_ms,
            "replace_id": replace_id,
        }
        if self._mode in {"try", "dbus"}:
            try:
                notice_id = self._notify_dbus(**kwargs)
            except Exception as exc:
                log.debug("dbus notification failed: %s", exc)
                self._drop_connection()
                if self._mode == "dbus":
                    self._mode = "try"
            else:
                self._mode = "dbus"
                return notice_id
        if self._mode in {"try", "send"}:
            try:
                notice_id = self._notify_send(**kwargs)
            except Exception as exc:
                log.warning("notification service is unavailable: %s", exc)
                self._mode = "off"
                return 0
            self._mode = "send"
            return notice_id
        if not self._gave_up:
            self._gave_up = True
            log.warning("notification service is unavailable; further notices are dropped")
        return 0

    def close(self) -> None:
        self._drop_connection()
        self._mode = "off"


def notification_service_available() -> tuple[bool, str]:
    """Doctor probe. Does not post a notification."""
    dbus_detail = ""
    try:
        from jeepney import DBusAddress, new_method_call
        from jeepney.io.blocking import open_dbus_connection

        connection = open_dbus_connection(bus="SESSION", auth_timeout=_SEND_TIMEOUT_S)
        try:
            message = new_method_call(
                DBusAddress(
                    "/org/freedesktop/DBus",
                    bus_name="org.freedesktop.DBus",
                    interface="org.freedesktop.DBus",
                ),
                "NameHasOwner",
                "s",
                ("org.freedesktop.Notifications",),
            )
            reply = connection.send_and_get_reply(message, timeout=_SEND_TIMEOUT_S)
            if bool(reply.body[0]):
                return True, "org.freedesktop.Notifications is owned"
            dbus_detail = "org.freedesktop.Notifications has no owner"
        finally:
            connection.close()
    except Exception as exc:
        dbus_detail = str(exc)
    if shutil.which("notify-send"):
        return True, "notify-send is on PATH"
    return False, dbus_detail or "notification service not found"
