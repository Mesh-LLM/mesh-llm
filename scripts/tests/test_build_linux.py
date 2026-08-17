#!/usr/bin/env python3
"""Focused argument-contract tests for scripts/build-linux.sh."""

from __future__ import annotations

import pathlib
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build-linux.sh"


class BuildLinuxArgumentTests(unittest.TestCase):
    def test_rejects_unknown_dash_prefixed_option(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--cuda-archh", "90"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown option: --cuda-archh", result.stderr)
        self.assertIn("usage: scripts/build-linux.sh", result.stderr)


if __name__ == "__main__":
    unittest.main()
