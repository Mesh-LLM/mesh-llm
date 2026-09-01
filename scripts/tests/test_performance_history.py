from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "performance-history.py"
SPEC = importlib.util.spec_from_file_location("performance_history", SCRIPT)
assert SPEC and SPEC.loader
HISTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HISTORY)


class PerformanceHistoryTests(unittest.TestCase):
    def _artifact(self, root: Path, throughput: float = 100.0) -> Path:
        artifact = root / "artifact"
        result_dir = artifact / "trace" / "cuda" / "qwen3" / "c-64" / "mesh"
        result_dir.mkdir(parents=True)
        provenance = artifact / "provenance"
        provenance.mkdir()
        cell = {
            "platform": "cuda",
            "model": "qwen3",
            "arm": "mesh",
            "workload": "thoughtworks",
            "concurrency": 64,
            "config_sha256": "c" * 64,
            "manifest_sha256": "p" * 64,
            "binary_sha256": "b" * 64,
            "comparison_capacity_policy": "matched",
        }
        (result_dir / "complete.json").write_text(
            json.dumps({"cell": cell, "cell_sha256": HISTORY.stable_hash(cell)}),
            encoding="utf-8",
        )
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "prompt_count": 64,
                    "successful_requests": 64,
                    "failed_requests": 0,
                    "output_tokens": 2048,
                    "measured_wall_ms": 20480.0,
                    "output_tokens_per_second": throughput,
                    "ttft_ms_mean": 500.0,
                }
            ),
            encoding="utf-8",
        )
        (provenance / "cuda.json").write_text(
            json.dumps(
                {
                    "created_utc": "2026-09-01T00:00:00Z",
                    "platform_details": "Linux-fixture",
                    "mesh_head": "a" * 40,
                    "native_runtime_directory_sha256": "n" * 64,
                    "models": {"qwen3": "m" * 64},
                }
            ),
            encoding="utf-8",
        )
        (artifact / "runner-gpu.csv").write_text(
            "RTX 5080,GPU-fixture,12.0,999.1,0000:01:00.0,P0,42,2400,12000\n",
            encoding="utf-8",
        )
        return artifact

    def test_normalize_records_reproducible_cohort_and_observed_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = HISTORY.normalize(self._artifact(Path(temp_dir)))
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertTrue(row["complete"])
        self.assertEqual("RTX 5080", row["cohort"]["hardware"]["name"])
        self.assertNotIn("uuid", row["cohort"]["hardware"])
        self.assertNotIn("pci_bus_id", row["cohort"]["hardware"])
        self.assertEqual(64, len(row["cohort"]["hardware"]["gpu_identity_sha256"]))
        self.assertEqual("42", row["observed_gpu_state"]["temperature_c"])
        self.assertEqual(64, row["cohort"]["concurrency"])
        self.assertEqual(64, len(row["cohort_key"]))

    def test_compare_requires_three_exact_cohort_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            current = HISTORY.normalize(self._artifact(Path(temp_dir), throughput=80.0))[0]
        baseline = []
        for index, throughput in enumerate((100.0, 101.0, 99.0)):
            row = json.loads(json.dumps(current))
            row["source_sha"] = str(index) * 40
            row["output_tokens_per_second"] = throughput
            baseline.append(row)
        comparison = HISTORY.compare([current], baseline, 0.05, 0.10)[0]
        self.assertEqual("performance-regression", comparison["classification"])
        different = json.loads(json.dumps(baseline[0]))
        different["cohort_key"] = "different"
        comparison = HISTORY.compare([current], [different] * 3, 0.05, 0.10)[0]
        self.assertEqual("insufficient-baseline", comparison["classification"])

    def test_history_directory_loads_immutable_run_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = {"schema_version": 1, "cohort_key": "fixture"}
            shard = root / "data" / "runs" / "abc" / "1.jsonl"
            shard.parent.mkdir(parents=True)
            shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual([row], HISTORY.read_jsonl(root))


if __name__ == "__main__":
    unittest.main()
