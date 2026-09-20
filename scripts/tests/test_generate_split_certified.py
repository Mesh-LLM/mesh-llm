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
    def test_patch_identity_covers_all_three_queue_lanes_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patch_root = Path(directory)
            model_support = patch_root / "model_support"
            generated = patch_root / "generated"
            model_support.mkdir()
            generated.mkdir()
            (patch_root / "0001-core.patch").write_bytes(b"core")
            (model_support / "0001-model.patch").write_bytes(b"model")
            (model_support / "series").write_text(
                "0001-model.patch\n", encoding="utf-8"
            )
            (generated / "0001-family-test.patch").write_bytes(b"generated")
            (generated / "series").write_text(
                "0001-family-test.patch\n", encoding="utf-8"
            )
            original_patch_dir = GENERATOR.PATCH_DIR
            try:
                GENERATOR.PATCH_DIR = patch_root
                self.assertEqual(
                    [
                        "0001-core.patch",
                        "model_support/0001-model.patch",
                        "generated/0001-family-test.patch",
                    ],
                    [
                        path.relative_to(patch_root).as_posix()
                        for path in GENERATOR.ordered_patch_queue()
                    ],
                )
                initial = GENERATOR.patch_queue_sha256()
                (model_support / "0001-model.patch").write_bytes(b"model changed")
                self.assertNotEqual(initial, GENERATOR.patch_queue_sha256())
            finally:
                GENERATOR.PATCH_DIR = original_patch_dir

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
