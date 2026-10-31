"""Dispatcher coalescing, ordering, and shutdown."""

from __future__ import annotations

import logging
import threading
import time

from gesture_media.action_types import Action, Kind
from gesture_media.actions import ActionDispatcher
from gesture_media.players.base import Capability
from gesture_media.players.fake import FakeAudio, FakePlayer


def test_coalesce_sums_volume_and_seek_and_keeps_last_pitch() -> None:
    seen: list[Action] = []
    caller = threading.current_thread()
    threads: list[threading.Thread] = []

    def handler(action: Action) -> None:
        threads.append(threading.current_thread())
        seen.append(action)

    dispatcher = ActionDispatcher(handler)
    dispatcher.submit_many(
        [
            Action(Kind.PLAY_PAUSE, "Open_Palm"),
            Action(Kind.VOLUME, "Pointing_Up", 1.0),
            Action(Kind.NEXT, "Thumb_Up"),
            Action(Kind.VOLUME, "Pointing_Up", 2.5),
            Action(Kind.SEEK, "Victory", 0.5),
            Action(Kind.PITCH, "ILoveYou", 1.0),
            Action(Kind.PREVIOUS, "Thumb_Down"),
            Action(Kind.PITCH, "ILoveYou", 2.5),
            Action(Kind.SEEK, "Victory", -0.25),
            Action(Kind.MUTE_TOGGLE, "Closed_Fist"),
        ]
    )
    dispatcher.shutdown()

    assert [action.kind for action in seen] == [
        Kind.PLAY_PAUSE,
        Kind.NEXT,
        Kind.VOLUME,
        Kind.PREVIOUS,
        Kind.PITCH,
        Kind.SEEK,
        Kind.MUTE_TOGGLE,
    ]
    volume = next(action for action in seen if action.kind is Kind.VOLUME)
    seek = next(action for action in seen if action.kind is Kind.SEEK)
    pitch = next(action for action in seen if action.kind is Kind.PITCH)
    assert volume.value == 3.5
    assert volume.gesture == "Pointing_Up"
    assert seek.value == 0.25
    assert pitch.value == 2.5
    assert pitch.gesture == "ILoveYou"
    assert threads
    assert all(thread is not caller for thread in threads)


def test_handler_exception_is_logged_and_worker_continues(caplog) -> None:
    seen: list[Kind] = []

    def handler(action: Action) -> None:
        if action.kind is Kind.NEXT:
            raise RuntimeError("nope")
        seen.append(action.kind)

    dispatcher = ActionDispatcher(handler)
    with caplog.at_level(logging.ERROR):
        dispatcher.submit_many(
            [
                Action(Kind.NEXT, "Thumb_Up"),
                Action(Kind.PLAY_PAUSE, "Open_Palm"),
            ]
        )
        dispatcher.shutdown()
    assert seen == [Kind.PLAY_PAUSE]
    assert "nope" in caplog.text


def test_shutdown_drains_pending_work() -> None:
    seen: list[Kind] = []

    def handler(action: Action) -> None:
        time.sleep(0.02)
        seen.append(action.kind)

    dispatcher = ActionDispatcher(handler)
    dispatcher.submit(Action(Kind.NEXT, "Thumb_Up"))
    dispatcher.submit(Action(Kind.PREVIOUS, "Thumb_Down"))
    dispatcher.submit(Action(Kind.MUTE_TOGGLE, "Closed_Fist"))
    dispatcher.shutdown()
    assert seen == [Kind.NEXT, Kind.PREVIOUS, Kind.MUTE_TOGGLE]


def test_fake_player_and_audio_record_calls() -> None:
    player = FakePlayer()
    assert player.name == "fake"
    assert set(player.capabilities) == set(Capability)
    assert player.is_available() is True
    player.play_pause()
    player.next()
    player.previous()
    player.open()
    player.close()
    player.close()
    player.seek(10)
    player.seek(-3)
    player.set_pitch(2.5)
    assert player.get_position() == 7
    assert player.pitch == 2.5
    assert player.calls == [
        ("is_available",),
        ("play_pause",),
        ("next",),
        ("previous",),
        ("open",),
        ("close",),
        ("close",),
        ("seek", 10),
        ("seek", -3),
        ("set_pitch", 2.5),
        ("get_position",),
    ]

    audio = FakeAudio()
    assert audio.get_state() == (50.0, False)
    audio.volume_delta(60)
    assert audio.get_state() == (100.0, False)
    audio.volume_delta(-250)
    assert audio.get_state() == (0.0, False)
    audio.toggle_mute()
    audio.toggle_mute()
    assert audio.get_state() == (0.0, False)
    assert ("volume_delta", 60) in audio.calls
    assert audio.calls.count(("toggle_mute",)) == 2
