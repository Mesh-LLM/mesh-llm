#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate-split-certified.py"
SPEC = importlib.util.spec_from_file_location("generate_split_certified", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class SplitCertificationRosterTests(unittest.TestCase):
    def test_non_chat_workload_evidence_never_grants_split_admission(self) -> None:
        """Only architectures backed by causal split evidence enter the roster."""
        manifest = json.loads(GENERATOR.DEFAULT_MANIFEST.read_text())
        roster = GENERATOR.build_roster(manifest)
        causal = [model for model in manifest["models"]
                  if model["class"] == "causal_generation"]
        self.assertEqual(83, len(causal))
        self.assertEqual(
            {model["architecture"] for model in causal}, set(roster["architectures"])
        )
        for model in manifest["models"]:
            if model["class"] != "causal_generation":
                model["architecture"] = f"non-chat-{model['class']}"
        self.assertEqual(roster, GENERATOR.build_roster(manifest))

    def test_workload_evidence_cannot_supply_the_only_split_architecture(self) -> None:
        """A shared architecture label does not promote non-chat evidence to split evidence."""
        manifest = json.loads(GENERATOR.DEFAULT_MANIFEST.read_text())
        manifest["models"] = [model for model in manifest["models"]
                              if model["class"] == "speech_recognition"]
        self.assertEqual("llama", manifest["models"][0]["architecture"])
        with self.assertRaisesRegex(GENERATOR.RosterError, "no split-certified architectures"):
            GENERATOR.build_roster(manifest)

    def test_invalid_workload_class_or_profile_cannot_grant_split_admission(self) -> None:
        """Fail closed on absent classes or non-chat rows mislabeled as split-certified."""
        for fields in ({"class": None}, {"class": "future"},
                       {"class": "embedding", "profile": "full"},
                       {"class": "causal_generation", "profile": "workload-oracle"}):
            with self.subTest(fields=fields):
                manifest = json.loads(GENERATOR.DEFAULT_MANIFEST.read_text())
                manifest["models"][0].update(fields)
                with self.assertRaises(GENERATOR.RosterError):
                    GENERATOR.build_roster(manifest)

    def test_roster_contains_unique_tested_architectures(self) -> None:
        """Keep the versioned architecture roster sorted, unique and artifact-independent."""
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
