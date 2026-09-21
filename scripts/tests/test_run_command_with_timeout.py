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
    def test_deadline_observes_child_exit_before_declaring_timeout(self) -> None:
        """An exit between the last wait and the deadline retains its actual status."""
        for child_status in (0, 7, None):
            with self.subTest(child_status=child_status):
                process = mock.Mock()
                process.wait.side_effect = subprocess.TimeoutExpired("fixture", 0.1)
                process.poll.return_value = child_status
                args = argparse.Namespace(seconds=1, label="fixture", command=["fixture"])
                with (
                    mock.patch.object(RUNNER, "parse_args", return_value=args),
                    mock.patch.object(RUNNER.signal, "signal"),
                    mock.patch.object(RUNNER.subprocess, "Popen", return_value=process),
                    mock.patch.object(RUNNER.time, "monotonic", side_effect=[0, 0.5, 1]),
                    mock.patch.object(RUNNER, "terminate_group") as terminate,
                ):
                    self.assertEqual(124 if child_status is None else child_status, RUNNER.main())
                process.wait.assert_called_once_with(timeout=0.1)
                process.poll.assert_called_once_with()
                if child_status is None:
                    terminate.assert_called_once_with(process)
                else:
                    terminate.assert_not_called()

    def exercise_signal_boundary(self, boundary: str, signum: int) -> None:
        """Inject cancellation at a boundary without sending signals to the test runner."""
        handlers = {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}
        events: list[str] = []
        in_spawn = False
        in_wait = False
        process = mock.Mock()

        def install_handler(number, handler):
            """Track handler replacement without changing the test process's real signals."""
            previous = handlers[number]
            handlers[number] = handler
            return previous

        def request_cancel(number):
            """Deliver a synthetic signal through the installed wrapper handler."""
            self.assertTrue(callable(handlers[number]), "handler must be installed before spawn")
            handlers[number](number, None)

        def spawn(*_args, **_kwargs):
            """Expose the interval before Popen returns ownership of the child."""
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
            """Interrupt a simulated wait while its internal lock is still held."""
            nonlocal in_wait
            in_wait = True
            try:
                request_cancel(signum)
                raise subprocess.TimeoutExpired("fixture", kwargs["timeout"])
            finally:
                in_wait = False

        def cleanup(actual_process):
            """Prove cleanup is deferred and repeated signals cannot reenter it."""
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
        """A signal arriving during spawn must not orphan the newly created process."""
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                self.exercise_signal_boundary("spawn", signum)

    def test_cancellation_during_wait_defers_cleanup_until_wait_unwinds(self) -> None:
        """A signal arriving during wait must not deadlock by recursively waiting."""
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                self.exercise_signal_boundary("wait", signum)


if __name__ == "__main__":
    unittest.main()
