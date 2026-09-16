#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate-split-certified.py"
SPEC = importlib.util.spec_from_file_location("generate_split_certified", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class SplitCertificationRosterTests(unittest.TestCase):
    def test_roster_contains_unique_tested_architectures(self) -> None:
        manifest = json.loads(GENERATOR.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        roster = GENERATOR.build_roster(manifest)
        self.assertEqual(2, roster["schema_version"])
        self.assertEqual(sorted(set(roster["architectures"])), roster["architectures"])
        self.assertIn("inkling", roster["architectures"])
        self.assertNotIn("models", roster)

    def test_checked_in_roster_is_deterministic_and_current(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_check_rejects_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split-certified.json"
            output.write_text(json.dumps({"schema_version": 0}), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output",
                    str(output),
                    "--check",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(1, result.returncode)
            self.assertIn("is stale", result.stderr)


if __name__ == "__main__":
    unittest.main()
