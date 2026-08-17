import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "ci-client-readiness-smoke.sh"


def write_runtime(path: pathlib.Path, signal_body: str) -> None:
    handler = textwrap.indent(textwrap.dedent(signal_body).strip(), "    ")
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import signal
import time

marker = os.environ["SMOKE_MARKER"]
with open(marker, "w", encoding="utf-8") as fh:
    fh.write(f"start:{{os.getpid()}}\\n")

def stop(signum, frame):
    del frame
    signal_name = signal.Signals(signum).name
{handler}

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
print(json.dumps({{"message": "Client ready"}}), flush=True)
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


class CiClientReadinessSmokeTests(unittest.TestCase):
    def test_product_readiness_is_hermetic(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("client --mesh-discovery-mode mdns", script)
        self.assertNotIn("client --auto", script)

    def run_smoke(
        self, runtime: pathlib.Path, root: pathlib.Path
    ) -> subprocess.CompletedProcess[str]:
        runtime_root = root / "native-runtimes"
        state_parent = root / "state"
        runtime_root.mkdir()
        state_parent.mkdir()
        env = os.environ.copy()
        env.update(
            {
                "MESH_LLM_CLIENT_READY_MAX_WAIT": "3",
                "MESH_LLM_CLIENT_SHUTDOWN_MAX_WAIT": "1",
                "MESH_LLM_CLIENT_STATE_PARENT": str(state_parent),
                "SMOKE_MARKER": str(root / "events"),
            }
        )
        return subprocess.run(
            ["bash", str(SCRIPT), str(runtime), str(runtime_root)],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=10,
        )

    def assert_process_absent(self, marker: pathlib.Path) -> None:
        pid = int(marker.read_text(encoding="utf-8").splitlines()[0].split(":")[1])
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotIn(f'","{pid}",', result.stdout)
            return
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_clean_sigterm_shutdown_succeeds_and_reaps_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            runtime = root / "mesh-llm"
            write_runtime(
                runtime,
                """\
                with open(marker, "a", encoding="utf-8") as fh:
                    fh.write(f"signal:{signal_name}:{os.getpid()}\\n")
                raise SystemExit(0)
                """,
            )

            result = self.run_smoke(runtime, root)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            marker = root / "events"
            pids = [
                int(line.rsplit(":", 1)[1])
                for line in marker.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(pids, [pids[0], pids[0]])
            self.assertIn(
                f"signal:SIGTERM:{pids[0]}",
                marker.read_text(encoding="utf-8").splitlines(),
            )
            self.assert_process_absent(marker)
            self.assertEqual(list((root / "state").iterdir()), [])

    def test_exit_cleanup_failure_overrides_ready_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            runtime = root / "mesh-llm"
            write_runtime(runtime, "return")

            result = self.run_smoke(runtime, root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "client did not stop cleanly after SIGTERM within 1s", result.stderr
            )
            self.assert_process_absent(root / "events")
            self.assertEqual(list((root / "state").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
