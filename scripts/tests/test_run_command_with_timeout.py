"""Deterministic signal-boundary regressions for the canary process supervisor."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import signal
import subprocess
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "run-command-with-timeout.py"
SPEC = importlib.util.spec_from_file_location("timeout_runner", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TimeoutSignalSafetyTests(unittest.TestCase):
    def exercise_signal_boundary(self, boundary: str, signum: int) -> None:
        """Inject cancellation at a boundary without sending signals to the test runner."""
        handlers = {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}
        events: list[str] = []
        in_spawn = False
        in_wait = False
        process = mock.Mock()

        def install_handler(number, handler):
            previous = handlers[number]
            handlers[number] = handler
            return previous

        def request_cancel(number):
            self.assertTrue(callable(handlers[number]), "handler must be installed before spawn")
            handlers[number](number, None)

        def spawn(*_args, **_kwargs):
            nonlocal in_spawn
            in_spawn = True
            try:
                if boundary == "spawn":
                    request_cancel(signum)
            finally:
                in_spawn = False
            events.append("spawned")
            return process

        def wait(*_args, **kwargs):
            nonlocal in_wait
            in_wait = True
            try:
                request_cancel(signum)
                raise subprocess.TimeoutExpired("fixture", kwargs["timeout"])
            finally:
                in_wait = False

        def cleanup(actual_process):
            self.assertIs(process, actual_process)
            self.assertFalse(in_spawn, "cleanup needs the completed Popen object")
            self.assertFalse(in_wait, "cleanup must not reenter Popen.wait from a signal handler")
            # Repeated cancellation during cleanup must neither reenter cleanup
            # nor replace the signal that determined the wrapper's exit status.
            request_cancel(signal.SIGINT if signum == signal.SIGTERM else signal.SIGTERM)
            events.append("cleaned")

        process.wait.side_effect = wait
        args = argparse.Namespace(seconds=30, label="fixture", command=["fixture"])
        with (
            mock.patch.object(RUNNER, "parse_args", return_value=args),
            mock.patch.object(RUNNER.signal, "signal", side_effect=install_handler),
            mock.patch.object(RUNNER.subprocess, "Popen", side_effect=spawn) as popen,
            mock.patch.object(RUNNER, "terminate_group", side_effect=cleanup) as terminate,
        ):
            self.assertEqual(128 + signum, RUNNER.main())
        popen.assert_called_once_with(
            ["fixture"], stdin=subprocess.DEVNULL, start_new_session=True,
        )
        terminate.assert_called_once_with(process)
        self.assertEqual(["spawned", "cleaned"], events)
        self.assertEqual({signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}, handlers)

    def test_cancellation_during_spawn_waits_for_child_ownership(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                self.exercise_signal_boundary("spawn", signum)

    def test_cancellation_during_wait_defers_cleanup_until_wait_unwinds(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                self.exercise_signal_boundary("wait", signum)


if __name__ == "__main__":
    unittest.main()
