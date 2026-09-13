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
    def test_single_file_identity_is_exact_blob_digest(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            digest,
            GENERATOR.aggregate_source_sha256(
                ["model.gguf"],
                {"model.gguf": {"size_bytes": 42, "blob_id": digest}},
            ),
        )

    def test_multi_file_identity_is_ordered_and_path_independent(self) -> None:
        integrity = {
            "one.gguf": {"size_bytes": 10, "blob_id": "a" * 64},
            "two.gguf": {"size_bytes": 20, "blob_id": "b" * 64},
        }
        first = GENERATOR.aggregate_source_sha256(
            ["one.gguf", "two.gguf"], integrity
        )
        relocated = GENERATOR.aggregate_source_sha256(
            ["nested/one.gguf", "nested/two.gguf"],
            {
                "nested/one.gguf": integrity["one.gguf"],
                "nested/two.gguf": integrity["two.gguf"],
            },
        )
        reversed_digest = GENERATOR.aggregate_source_sha256(
            ["two.gguf", "one.gguf"], integrity
        )
        self.assertEqual(first, relocated)
        self.assertNotEqual(first, reversed_digest)

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
