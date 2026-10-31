"""Background action dispatcher.

The caller submits actions. A worker thread runs the handler so subprocess
work never blocks the frame loop. Continuous actions drained in one batch are
collapsed: volume deltas sum, seek deltas sum, and the last pitch wins.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable

from gesture_media.action_types import Action, Kind

logger = logging.getLogger(__name__)


def coalesce_actions(actions: Iterable[Action]) -> list[Action]:
    """Collapse one drained batch.

    Discrete actions stay in order. Each continuous kind is emitted once, at
    the index of its last occurrence.
    """
    items = list(actions)
    volume_sum = 0.0
    seek_sum = 0.0
    volume_at: int | None = None
    seek_at: int | None = None
    pitch_at: int | None = None
    volume_gesture = ""
    seek_gesture = ""
    pitch_action: Action | None = None
    for index, action in enumerate(items):
        if action.kind is Kind.VOLUME and action.value is not None:
            volume_sum += action.value
            volume_at = index
            volume_gesture = action.gesture
        elif action.kind is Kind.SEEK and action.value is not None:
            seek_sum += action.value
            seek_at = index
            seek_gesture = action.gesture
        elif action.kind is Kind.PITCH:
            pitch_at = index
            pitch_action = action
    collapsed: list[Action] = []
    for index, action in enumerate(items):
        if action.kind is Kind.VOLUME:
            if index == volume_at:
                collapsed.append(Action(Kind.VOLUME, volume_gesture, volume_sum))
        elif action.kind is Kind.SEEK:
            if index == seek_at:
                collapsed.append(Action(Kind.SEEK, seek_gesture, seek_sum))
        elif action.kind is Kind.PITCH:
            if index == pitch_at and pitch_action is not None:
                collapsed.append(pitch_action)
        else:
            collapsed.append(action)
    return collapsed


class ActionDispatcher:
    """Queue actions onto one worker thread and coalesce each drained batch."""

    def __init__(self, handler: Callable[[Action], None]) -> None:
        self._handler = handler
        self._pending: list[Action] = []
        self._closed = False
        self._cv = threading.Condition()
        self._thread = threading.Thread(
            target=self._worker,
            name="gesture-action-dispatcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, action: Action) -> None:
        """Enqueue one action. Returns without running the handler."""
        self.submit_many((action,))

    def submit_many(self, actions: Iterable[Action]) -> None:
        """Enqueue actions as one batch if the worker has not taken them yet."""
        with self._cv:
            if self._closed:
                raise RuntimeError("dispatcher is shut down")
            self._pending.extend(actions)
            self._cv.notify()

    def shutdown(self) -> None:
        """Drain queued actions, then join the worker."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join()

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                batch = self._pending
                self._pending = []
                closed = self._closed
            if batch:
                self._run_batch(batch)
            if closed:
                return

    def _run_batch(self, batch: list[Action]) -> None:
        for action in coalesce_actions(batch):
            try:
                self._handler(action)
            except Exception:
                logger.exception("action handler failed for %s", action.kind)
