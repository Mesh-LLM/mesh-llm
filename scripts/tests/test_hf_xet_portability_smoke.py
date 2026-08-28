#!/usr/bin/env python3
"""Hermetic contracts for the release-artifact Hugging Face/Xet smoke."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci-hf-xet-portability-smoke.sh"


class HfXetPortabilitySmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        for tool in ("cat", "mkdir", "mktemp", "python3", "rm", "uname"):
            target = shutil.which(tool)
            if target is None:
                self.fail(f"required test tool is unavailable: {tool}")
            (self.bin_dir / tool).symlink_to(target)
        self.write_executable("sleep", "#!/bin/sh\nexit 0\n")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_executable(self, name: str, source: str) -> Path:
        path = self.bin_dir / name
        path.write_text(source, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def run_smoke(self, binary: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PATH"] = str(self.bin_dir)
        return subprocess.run(
            ["/bin/bash", str(SCRIPT), str(binary)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_missing_timeout_is_an_advisory_skip(self) -> None:
        binary = self.write_executable("mesh-llm", "#!/bin/sh\nexit 0\n")

        result = self.run_smoke(binary)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("advisory smoke skipped", result.stderr)

    def test_transient_download_failures_are_retried(self) -> None:
        self.write_executable("timeout", "#!/bin/sh\nshift\nexec \"$@\"\n")
        binary = self.write_executable(
            "mesh-llm",
            """#!/bin/sh
count_file="$MESH_LLM_DATA_DIR/attempts"
count=0
if [ -f "$count_file" ]; then count="$(sed -n '1p' "$count_file")"; fi
count=$((count + 1))
printf '%s\\n' "$count" >"$count_file"
if [ "$count" -lt 3 ]; then echo transient >&2; exit 1; fi
fixture="$HF_HOME/fixture.gguf"
printf 'fixture' >"$fixture"
printf '{"path":"%s"}\\n' "$fixture"
""",
        )
        sed = shutil.which("sed")
        if sed is None:
            self.fail("required test tool is unavailable: sed")
        (self.bin_dir / "sed").symlink_to(sed)

        result = self.run_smoke(binary)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("attempt 3/3", result.stderr)
        self.assertIn("Xet portability smoke passed", result.stdout)

    def test_sigill_remains_a_hard_failure(self) -> None:
        self.write_executable("timeout", "#!/bin/sh\nshift\nexec \"$@\"\n")
        binary = self.write_executable("mesh-llm", "#!/bin/sh\nkill -ILL $$\n")

        result = self.run_smoke(binary)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SIGILL", result.stderr)


if __name__ == "__main__":
    unittest.main()
