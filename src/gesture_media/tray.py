"""StatusNotifierItem tray via pystray. A missing tray host does not stop the app."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

_COLORS = {
    "active": (40, 170, 70, 255),
    "paused": (214, 158, 32, 255),
    "error": (196, 48, 40, 255),
}


def tray_available() -> tuple[bool, str]:
    try:
        import pystray  # noqa: F401
    except ImportError:
        return False, "pystray is not installed"
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False, "no DISPLAY or WAYLAND_DISPLAY; tray disabled"
    return True, "pystray is installed and a display is set (StatusNotifierItem / AppIndicator)"


def draw_icon(state: str):
    from PIL import Image, ImageDraw

    color = _COLORS.get(state, _COLORS["error"])
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((4, 4, 60, 60), fill=color)
    draw.polygon([(26, 18), (26, 46), (46, 32)], fill=(255, 255, 255, 255))
    return image


def tooltip(state: str, backend_name: str) -> str:
    return f"Gesture Media Controller — {state} ({backend_name})"


@dataclass
class TrayHooks:
    toggle_pause: Callable[[], None]
    toggle_preview: Callable[[], None]
    calibrate: Callable[[], None]
    open_config: Callable[[], None]
    set_notifications: Callable[[str], None]
    toggle_live_updates: Callable[[], None]
    toggle_final_value: Callable[[], None]
    quit: Callable[[], None]
    paused: Callable[[], bool]
    preview_visible: Callable[[], bool]
    notification_mode: Callable[[], str]
    live_updates: Callable[[], bool]
    final_value: Callable[[], bool]


class TrayIcon:
    def __init__(self, icon, backend_name: str) -> None:
        self._icon = icon
        self.backend_name = backend_name
        self.state = "active"

    def set_state(self, state: str) -> None:
        if state not in _COLORS:
            state = "error"
        self.state = state
        try:
            self._icon.icon = draw_icon(state)
            self._icon.title = tooltip(state, self.backend_name)
        except Exception as exc:
            log.warning("could not update the tray icon: %s", exc)

    def close(self) -> None:
        try:
            self._icon.visible = False
            self._icon.stop()
        except Exception:
            log.debug("tray stop failed", exc_info=True)


def _menu(pystray, hooks: TrayHooks):
    def pause_label(_item) -> str:
        return "Resume detection" if hooks.paused() else "Pause detection"

    def preview_label(_item) -> str:
        return "Hide preview" if hooks.preview_visible() else "Show preview"

    def mode_checked(mode: str):
        return lambda _item: hooks.notification_mode() == mode

    notifications = pystray.Menu(
        pystray.MenuItem(
            "All",
            lambda: hooks.set_notifications("all"),
            checked=mode_checked("all"),
        ),
        pystray.MenuItem(
            "Important only",
            lambda: hooks.set_notifications("important"),
            checked=mode_checked("important"),
        ),
        pystray.MenuItem(
            "Off",
            lambda: hooks.set_notifications("off"),
            checked=mode_checked("off"),
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "Live updates",
            lambda: hooks.toggle_live_updates(),
            checked=lambda _item: hooks.live_updates(),
        ),
        pystray.MenuItem(
            "Final value",
            lambda: hooks.toggle_final_value(),
            checked=lambda _item: hooks.final_value(),
        ),
    )
    return pystray.Menu(
        pystray.MenuItem(pause_label, lambda: hooks.toggle_pause()),
        pystray.MenuItem(preview_label, lambda: hooks.toggle_preview()),
        pystray.MenuItem("Calibrate", lambda: hooks.calibrate()),
        pystray.MenuItem("Open config file", lambda: hooks.open_config()),
        pystray.MenuItem("Notifications", notifications),
        pystray.MenuItem("Quit", lambda: hooks.quit()),
    )


def start_tray(hooks: TrayHooks, *, backend_name: str) -> TrayIcon | None:
    """Return a running tray, or None when the host or the display is missing."""
    ok, detail = tray_available()
    if not ok:
        log.warning("%s The app will keep running without a tray.", detail)
        return None
    try:
        import pystray
    except ImportError as exc:
        log.warning("Tray icon is unavailable (%s). The app will keep running.", exc)
        return None
    try:
        icon = pystray.Icon(
            "gesture-media",
            draw_icon("active"),
            tooltip("active", backend_name),
            menu=_menu(pystray, hooks),
        )
        icon.run_detached()
    except Exception as exc:
        log.warning(
            "Tray icon is unavailable (%s). The app will keep running without a tray. "
            "KDE Plasma provides StatusNotifierItem; GNOME needs the AppIndicator extension.",
            exc,
        )
        return None
    return TrayIcon(icon, backend_name)
