from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO / "evals/skippy-scheduler-fixtures.py"
    spec = importlib.util.spec_from_file_location("skippy_scheduler_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module()
CATALOG_PATH = REPO / "evals/skippy-scheduler-fixtures.json"


class SchedulerFixturesTest(unittest.TestCase):
    def test_checked_in_catalog_is_valid_and_pins_measured_rows(self) -> None:
        catalog = FIXTURES.load_catalog(CATALOG_PATH)
        profile = catalog["profiles"]["agentic-eviction-pressure"]

        self.assertEqual(profile["workload"]["families"], 8)
        self.assertEqual(profile["workload"]["ctx_size"], 131072)
        self.assertEqual(len(profile["corpus"]["rows"]), 8)
        self.assertEqual(
            profile["corpus"]["prompt_manifest_sha256"],
            "f1ddbe3d5974f3f4bd06f5d70fa45d0e10305bbafa4eb7399a0f972458d1beef",
        )

    def test_fetch_uses_pinned_hf_cache_and_strict_verification(self) -> None:
        catalog = FIXTURES.load_catalog(CATALOG_PATH)
        dataset = catalog["datasets"]["agentic-coding-trajectories"]
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            snapshot.mkdir()
            (snapshot / "sessions.parquet").touch()

            def runner(command, **kwargs):
                commands.append(command)
                stdout = json.dumps({"path": str(snapshot)}) if command[1] == "download" else "{}"
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

            parquet = FIXTURES.fetch_dataset(dataset, Path(directory) / "cache", runner)

        self.assertEqual(parquet.name, "sessions.parquet")
        self.assertEqual(commands[0][:3], ["hf", "download", dataset["repo_id"]])
        self.assertIn(dataset["revision"], commands[0])
        self.assertEqual(commands[1][:3], ["hf", "cache", "verify"])
        self.assertIn("--fail-on-missing-files", commands[1])

    def test_materialization_rejects_row_provenance_drift(self) -> None:
        catalog = FIXTURES.load_catalog(CATALOG_PATH)
        selected = catalog["profiles"]["agentic-eviction-pressure"]
        row = {**selected["corpus"]["rows"][0], "messages_json": "[]"}
        row["session_id"] = "unexpected"
        generator = SimpleNamespace(select_trajectories=lambda *_args: [row] * 8)
        with tempfile.TemporaryDirectory() as directory:
            parquet = Path(directory) / "sessions.parquet"
            parquet.touch()
            output = Path(directory) / "prompts.json"
            with mock.patch.object(FIXTURES, "load_generator", return_value=generator):
                with self.assertRaisesRegex(RuntimeError, "pinned fixture provenance"):
                    FIXTURES.materialize_prompt_manifest(catalog, selected, parquet, output)

    def test_manifest_hash_failure_preserves_an_existing_output(self) -> None:
        catalog = FIXTURES.load_catalog(CATALOG_PATH)
        selected = catalog["profiles"]["agentic-eviction-pressure"]
        rows = [
            {**row, "messages_json": '[{"role": "user", "content": "fixture"}]'}
            for row in selected["corpus"]["rows"]
        ]
        generator = SimpleNamespace(
            select_trajectories=lambda *_args: rows,
            build_manifest=lambda *_args: {"unexpected": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            parquet = Path(directory) / "sessions.parquet"
            parquet.touch()
            output = Path(directory) / "prompts.json"
            output.write_text("keep-me")
            with mock.patch.object(FIXTURES, "load_generator", return_value=generator):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    FIXTURES.materialize_prompt_manifest(catalog, selected, parquet, output)

            self.assertEqual(output.read_text(), "keep-me")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
