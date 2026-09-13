from __future__ import annotations

import os
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check-skippy-workload-candidate.py"
SPEC = importlib.util.spec_from_file_location("workload_candidate", CHECK)
assert SPEC and SPEC.loader
CANDIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CANDIDATE)


class CandidateBuildFreshnessTests(unittest.TestCase):
    def test_producer_binds_source_native_stamp_and_every_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            native = root / "native"
            (native / "bin").mkdir(parents=True)
            stamp = native / ".mesh-llm-build-stamp"
            stamp.write_text("cpu native fixture")
            os.utime(stamp, ns=(1, 1))
            binary, test_binary = root / "skippy-server", root / "skippy-tests"
            files = CANDIDATE.producer_files(binary, native, test_binary)
            for name, path in files.items():
                if name != "native_stamp":
                    path.write_text(name)
                    path.chmod(0o755)
            source = {"head": "a" * 40, "worktree_sha256": "b" * 64}
            snapshot = root / "source.json"
            snapshot.write_text(json.dumps(source))
            manifest = root / "producer.json"
            with mock.patch.object(CANDIDATE, "source_identity", return_value=source):
                CANDIDATE.write_producer(manifest, binary, native, test_binary, snapshot)
                CANDIDATE.verify_producer(manifest, binary, native)
                for name, path in files.items():
                    with self.subTest(artifact=name):
                        original = path.read_bytes()
                        path.write_bytes(original + b"tampered")
                        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                            CANDIDATE.verify_producer(manifest, binary, native)
                        path.write_bytes(original)
                source["head"] = "c" * 40
                with self.assertRaisesRegex(RuntimeError, "current repository head"):
                    CANDIDATE.verify_producer(manifest, binary, native)
                with self.assertRaisesRegex(RuntimeError, "source changed while building"):
                    CANDIDATE.write_producer(manifest, binary, native, test_binary, snapshot)

    def _check(self, binary: Path, build_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CHECK),
                "--candidate-binary",
                str(binary),
                "--native-build-dir",
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_accepts_binary_linked_after_stamped_native_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            stamp = directory / ".mesh-llm-build-stamp"
            binary = directory / "skippy-server"
            stamp.touch()
            binary.touch(mode=0o755)
            os.utime(stamp, ns=(1_000_000_000, 1_000_000_000))
            os.utime(binary, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(0, self._check(binary, directory).returncode)

    def test_rejects_binary_older_than_or_equal_to_native_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            stamp = directory / ".mesh-llm-build-stamp"
            binary = directory / "skippy-server"
            stamp.touch()
            binary.touch(mode=0o755)
            os.utime(stamp, ns=(2_000_000_000, 2_000_000_000))
            for binary_time in (1_000_000_000, 2_000_000_000):
                with self.subTest(binary_time=binary_time):
                    os.utime(binary, ns=(binary_time, binary_time))
                    result = self._check(binary, directory)
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn("candidate executable predates", result.stderr)

    def test_rejects_missing_executable_or_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            binary = directory / "skippy-server"
            self.assertIn("candidate executable is missing", self._check(binary, directory).stderr)
            binary.touch(mode=0o755)
            self.assertIn("native build stamp is missing", self._check(binary, directory).stderr)


if __name__ == "__main__":
    unittest.main()
