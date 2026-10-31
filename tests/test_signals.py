"""Ctrl-C has to end the process, not only set a flag the loop may never read."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap


def test_sigint_exits_even_while_the_main_thread_is_blocked() -> None:
    code = textwrap.dedent(
        """
        import time
        from threading import Event

        from gesture_media.app import _install_signals

        _install_signals(Event())
        print("ready", flush=True)
        time.sleep(30)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = proc.stdout.readline().strip() if proc.stdout is not None else ""
        assert ready == "ready"
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=4) == 130
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
