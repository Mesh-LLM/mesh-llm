import importlib.util
import pathlib
import signal
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).parents[1] / "ci-client-readiness-process.py"
SPEC = importlib.util.spec_from_file_location("ci_client_readiness_process", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESS)


class ClientReadinessProcessTests(unittest.TestCase):
    def test_windows_launch_uses_a_new_process_group_and_records_native_pid(self):
        child = mock.Mock(pid=4242)
        child.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pid_file = root / "child.pid"
            log_file = root / "client.log"
            with mock.patch.object(
                PROCESS.subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0x200,
                create=True,
            ), mock.patch.object(PROCESS.subprocess, "Popen", return_value=child) as popen:
                status = PROCESS.launch(
                    ["mesh-llm.exe", "client", "--auto"],
                    pid_file,
                    log_file,
                    is_windows=True,
                )
            recorded_pid = pid_file.read_text(encoding="utf-8")

        self.assertEqual(status, 0)
        self.assertEqual(recorded_pid, "4242\n")
        self.assertEqual(popen.call_args.kwargs["creationflags"], 0x200)
        self.assertEqual(popen.call_args.args[0], ["mesh-llm.exe", "client", "--auto"])

    def test_unix_launch_does_not_set_windows_creation_flags(self):
        child = mock.Mock(pid=4242)
        child.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with mock.patch.object(PROCESS.subprocess, "Popen", return_value=child) as popen:
                PROCESS.launch(
                    ["mesh-llm", "client", "--auto"],
                    root / "child.pid",
                    root / "client.log",
                    is_windows=False,
                )

        self.assertEqual(popen.call_args.kwargs["creationflags"], 0)

    def test_ctrl_break_targets_the_native_process_group(self):
        with mock.patch.object(signal, "CTRL_BREAK_EVENT", 1, create=True), mock.patch.object(
            PROCESS.os, "kill"
        ) as kill:
            PROCESS.request_ctrl_break(4242)

        kill.assert_called_once_with(4242, 1)


if __name__ == "__main__":
    unittest.main()
