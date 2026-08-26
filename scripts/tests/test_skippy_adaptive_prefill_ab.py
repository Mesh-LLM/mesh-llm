#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[2] / "evals/skippy-adaptive-prefill-ab.py"
    spec = importlib.util.spec_from_file_location("skippy_adaptive_prefill_ab", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


class AdaptivePrefillAbTests(unittest.TestCase):
    def test_prompt_manifest_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": {"dataset_revision": "pinned"},
                        "prompts": [
                            {
                                "family": "trajectory-1",
                                "bucket": "4k-8k",
                                "source_id": "session-1",
                                "prompt": "real agent trace",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            prompts, metadata = BENCH.read_prompt_manifest(path)

        self.assertEqual(metadata, {"dataset_revision": "pinned"})
        self.assertEqual(prompts[0]["family"], "trajectory-1")
        self.assertEqual(prompts[0]["bucket"], "4k-8k")
        self.assertEqual(prompts[0]["source_id"], "session-1")

    def test_prompt_manifest_rejects_missing_family(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps({"prompts": [{"prompt": "trace"}]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "nonempty family"):
                BENCH.read_prompt_manifest(path)

    def test_paired_intervals_are_deterministic(self):
        base = {
            "ttft_ms_p50": 1.0,
            "ttft_ms_p95": 2.0,
            "prefill_elapsed_ms_p95": 3.0,
            "makespan_ms": 4.0,
            "output_tokens_per_second": 5.0,
            "prefill_chunk_count_median": 6.0,
            "prefill_max_chunk_size_median": 7.0,
        }
        cells = []
        for round_index in range(1, 5):
            cells.extend(
                [
                    {"round": round_index, "version": "old", "summary": base},
                    {
                        "round": round_index,
                        "version": "new",
                        "summary": {key: value * 1.1 for key, value in base.items()},
                    },
                ]
            )

        first = BENCH.paired_intervals(cells, 4)
        second = BENCH.paired_intervals(cells, 4)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["ttft_ms_p95"]["median"], 10.0)
        self.assertEqual(len(first["ttft_ms_p95"]["round_deltas"]), 4)


if __name__ == "__main__":
    unittest.main()
