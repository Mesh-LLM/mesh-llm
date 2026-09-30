"""Cross-platform contracts for the Laya product smoke driver."""

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_laya_smoke", ROOT / "scripts" / "ci-laya-smoke.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class LayaSmokeTest(unittest.TestCase):
    def test_windows_stop_terminates_the_entire_process_tree(self):
        process = mock.Mock(pid=1234)
        process.poll.return_value = None

        with mock.patch.object(smoke.os, "name", "nt"), mock.patch.object(
            smoke.subprocess, "run"
        ) as run:
            smoke.stop(process)

        run.assert_called_once_with(
            ["taskkill", "/PID", "1234", "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        process.wait.assert_called_once_with(timeout=10)
        process.terminate.assert_not_called()

    def test_windows_stop_falls_back_when_taskkill_does_not_finish(self):
        process = mock.Mock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("taskkill", 10), None]

        with mock.patch.object(smoke.os, "name", "nt"), mock.patch.object(
            smoke.subprocess, "run"
        ):
            smoke.stop(process)

        process.kill.assert_called_once_with()
        self.assertEqual([mock.call(timeout=10), mock.call(timeout=10)], process.wait.call_args_list)


if __name__ == "__main__":
    unittest.main()
