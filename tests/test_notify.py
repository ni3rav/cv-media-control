"""Notification policy and a notifier that never raises. No D-Bus connection."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from gesture_media.notify import (
    DesktopNotifier,
    FakeNotifier,
    NoticeState,
    apply_notices,
    notification_allowed,
    plan_notifications,
    urgency_byte,
)


def _config(**overrides):
    notes = dict(
        enabled=True,
        verbosity="all",
        update_hz=5,
        expire_discrete_ms=3000,
        expire_continuous_ms=1500,
        live_updates=True,
        final_value=True,
        on_start=True,
        on_toggle=True,
        on_error=True,
        per_action={},
    )
    notes.update(overrides)
    return SimpleNamespace(notifications=SimpleNamespace(**notes))


def _plan(config, state, event, **kwargs):
    return plan_notifications(config, state, event, **kwargs)


def test_urgency_byte():
    assert urgency_byte("low") == 0
    assert urgency_byte("normal") == 1


def test_one_discrete_notification():
    state, planned = _plan(_config(), NoticeState(), "play_pause", gesture="Open_Palm")
    assert len(planned) == 1
    assert planned[0].body == "Play/Pause (Open palm)"
    assert planned[0].urgency == "low"
    assert planned[0].expire_ms == 3000
    assert planned[0].slot is None
    notifier = FakeNotifier()
    apply_notices(notifier, state, planned)
    assert notifier.calls[0]["replace_id"] is None
    assert notifier.calls[0]["id"] == 1


def test_continuous_updates_share_one_id_and_are_throttled():
    config = _config()
    state = NoticeState()
    notifier = FakeNotifier()
    sent = []
    for t_ms, value in ((0, 40), (100, 42), (199, 43), (200, 45)):
        state, planned = _plan(
            config,
            state,
            "volume",
            t_ms=t_ms,
            gesture="Pointing_Up",
            value=value,
            changed=True,
        )
        sent.append(planned)
        state = apply_notices(notifier, state, planned)
    assert [len(item) for item in sent] == [1, 0, 0, 1]
    assert sent[0][0].body == "Volume 40% (Pointing up)"
    assert sent[0][0].expire_ms == 1500
    assert sent[0][0].slot == "volume"
    assert notifier.calls[1]["replace_id"] == notifier.calls[0]["id"]
    assert notifier.calls[0]["id"] == notifier.calls[1]["id"]
    state, planned = _plan(
        config,
        state,
        "volume",
        t_ms=500,
        gesture="Pointing_Up",
        value=45,
        changed=True,
        ended=True,
    )
    state = apply_notices(notifier, state, planned)
    assert planned[0].body == "Volume set to 45%"
    assert planned[0].expire_ms == 3000
    assert planned[0].slot == "volume"
    assert notifier.calls[2]["replace_id"] == notifier.calls[0]["id"]


def test_no_final_when_the_value_did_not_change():
    config = _config()
    state, planned = _plan(
        config,
        NoticeState(),
        "volume",
        t_ms=0,
        gesture="Pointing_Up",
        value=40,
        changed=False,
    )
    assert planned
    state, planned = _plan(
        config,
        state,
        "volume",
        t_ms=400,
        value=40,
        changed=False,
        ended=True,
        gesture="Pointing_Up",
    )
    assert planned == ()


def test_final_still_sent_when_the_gesture_ends():
    config = _config()
    state, _planned = _plan(
        config,
        NoticeState(),
        "seek",
        t_ms=0,
        gesture="Victory",
        value=4,
        changed=True,
    )
    state, planned = _plan(
        config,
        state,
        "seek",
        t_ms=1000,
        gesture="Victory",
        value=14,
        changed=True,
        ended=True,
        position_s=92,
    )
    assert planned[0].body == "Seek total +14 s, now at 1:32"
    state, without = _plan(
        config,
        state,
        "seek",
        t_ms=0,
        value=14,
        changed=True,
        ended=True,
        gesture="Victory",
    )
    assert without[0].body == "Seek total +14 s"


def test_live_and_final_combinations():
    cases = (
        (True, True, True, True),
        (False, True, False, True),
        (True, False, True, False),
        (False, False, False, False),
    )
    for live, final, expect_live, expect_final in cases:
        config = _config(live_updates=live, final_value=final)
        state, live_planned = _plan(
            config,
            NoticeState(),
            "pitch",
            t_ms=0,
            gesture="ILoveYou",
            value=2.5,
            changed=True,
        )
        _state, final_planned = _plan(
            config,
            state,
            "pitch",
            t_ms=1000,
            gesture="ILoveYou",
            value=2.5,
            changed=True,
            ended=True,
        )
        assert bool(live_planned) is expect_live
        assert bool(final_planned) is expect_final
        if expect_live:
            assert live_planned[0].body == "Pitch +2.5 semitones (I love you)"
            assert live_planned[0].expire_ms == 1500
        if expect_final:
            assert final_planned[0].body == "Pitch set to +2.5 semitones"
            assert final_planned[0].expire_ms == 3000


def test_verbosity_filtering():
    assert notification_allowed(_config(verbosity="off"), "app_on") is False
    assert notification_allowed(_config(enabled=False), "error") is False
    important = _config(verbosity="important")
    assert notification_allowed(important, "app_on") is True
    assert notification_allowed(important, "pause") is True
    assert notification_allowed(important, "error") is True
    assert notification_allowed(important, "play_pause") is False
    assert notification_allowed(important, "volume") is False
    quiet_start = _config(on_start=False)
    assert notification_allowed(quiet_start, "app_on") is False
    assert notification_allowed(quiet_start, "play_pause") is True
    no_toggle = _config(on_toggle=False)
    assert notification_allowed(no_toggle, "pause") is False
    no_error = _config(on_error=False)
    assert notification_allowed(no_error, "error") is False
    per = _config(per_action={"volume": False})
    _state, planned = _plan(per, NoticeState(), "volume", t_ms=0, value=10, gesture="Pointing_Up")
    assert planned == ()
    _state, hidden = _plan(
        _config(verbosity="important"),
        NoticeState(),
        "volume",
        t_ms=0,
        value=10,
        gesture="Pointing_Up",
    )
    assert hidden == ()
    _state, important_start = _plan(
        _config(verbosity="important"),
        NoticeState(),
        "app_on",
        backend="mpris",
    )
    assert important_start[0].body.startswith("Gesture Media Controller started")
    allowed, start = _plan(_config(), NoticeState(), "app_on", backend="mpris")
    assert start[0].body == "Gesture Media Controller started. Active, backend: mpris"
    _state, stop = _plan(_config(), allowed, "app_off")
    assert stop[0].body == "Gesture Media Controller stopped"
    _state, error = _plan(_config(), NoticeState(), "error", detail="Camera lost")
    assert error[0].urgency == "normal"
    assert error[0].body == "Camera lost"


def test_unsupported_notice_once():
    state = NoticeState()
    state, first = _plan( _config(), state, "unsupported", capability="pitch", backend="mpris")
    state, second = _plan(_config(), state, "unsupported", capability="pitch", backend="mpris")
    state, other = _plan(_config(), state, "unsupported", capability="pitch", backend="mpv")
    assert first[0].body == "Pitch not supported on mpris"
    assert second == ()
    assert other[0].body == "Pitch not supported on mpv"
    _state, hidden = _plan(
        _config(verbosity="important"),
        NoticeState(),
        "unsupported",
        capability="pitch",
        backend="mpris",
    )
    assert hidden == ()


def test_start_stop_pause_and_mute_text():
    state, started = _plan(_config(), NoticeState(), "app_on", backend="mpv")
    _state, stopped = _plan(_config(), state, "app_off")
    assert "backend: mpv" in started[0].body
    assert stopped[0].body == "Gesture Media Controller stopped"
    _state, paused = _plan(_config(), NoticeState(), "pause", gesture="two_fists")
    assert paused[0].body == "Detection paused (two fists)"
    _state, resumed = _plan(_config(), NoticeState(), "resume")
    assert resumed[0].body == "Detection resumed"
    _state, muted = _plan(
        _config(),
        NoticeState(),
        "mute",
        gesture="Closed_Fist",
        muted=True,
    )
    assert muted[0].body == "Mute toggled (Fist). Muted"
    _state, nxt = _plan(_config(), NoticeState(), "next", gesture="Thumb_Up")
    assert nxt[0].body == "Next track (Thumb up)"
    _state, reset = _plan(_config(), NoticeState(), "pitch_reset")
    assert reset[0].body == "Pitch reset"


def test_fake_notifier_replace_id_and_close():
    notifier = FakeNotifier()
    first = notifier.notify("s", "one")
    second = notifier.notify("s", "two", replace_id=first)
    third = notifier.notify("s", "three")
    assert second == first
    assert third != first
    assert notifier.calls[1]["replace_id"] == first
    notifier.close()
    assert notifier.closed is True


def test_desktop_notifier_prefers_dbus_and_then_gives_up():
    notifier = DesktopNotifier()
    calls = {"dbus": 0, "send": 0}

    def dbus(**kwargs):
        calls["dbus"] += 1
        return 9

    def send(**kwargs):
        calls["send"] += 1
        raise AssertionError("notify-send should not run when dbus works")

    notifier._notify_dbus = dbus
    notifier._notify_send = send
    assert notifier.notify("Hello", "Body", replace_id=3) == 9
    assert calls == {"dbus": 1, "send": 0}

    failing = DesktopNotifier()
    attempts = {"n": 0}

    def boom(**kwargs):
        attempts["n"] += 1
        raise RuntimeError("down")

    failing._notify_dbus = boom
    failing._notify_send = boom
    assert failing.notify("a", "b") == 0
    assert failing.notify("a", "c") == 0
    assert attempts["n"] == 2


def test_notify_send_argv(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(argv, 0, stdout="15\n", stderr="")

    monkeypatch.setattr("gesture_media.notify.subprocess.run", fake_run)
    notifier = DesktopNotifier()
    assert notifier._notify_send(
        summary="Hello",
        body="Body",
        urgency="low",
        expire_ms=3000,
        replace_id=4,
    ) == 15
    argv = captured["argv"]
    assert argv[:2] == ["notify-send", "--print-id"]
    assert "--replace-id" in argv
    assert argv[argv.index("--replace-id") + 1] == "4"
    assert "--hint=boolean:suppress-sound:true" in argv
    assert "Hello" in argv and "Body" in argv
    assert captured["timeout"] == 0.5


def test_dbus_notify_message_serialises():
    from gesture_media.notify import _notify_message

    message = _notify_message("Hello", "Body", "normal", 1500, 7)
    blob = message.serialise(serial=1)
    assert isinstance(blob, bytes)
    assert len(blob) > 20
