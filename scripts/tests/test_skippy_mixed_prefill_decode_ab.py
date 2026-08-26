#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "evals/skippy-mixed-prefill-decode-ab.py"
    )
    spec = importlib.util.spec_from_file_location(
        "skippy_mixed_prefill_decode_ab", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


class MixedPrefillDecodeAbTests(unittest.TestCase):
    def test_stable_prompt_is_deterministic_and_nonempty(self):
        first = BENCH.stable_prompt(4, 2, "anchor")
        second = BENCH.stable_prompt(4, 2, "anchor")

        self.assertEqual(first, second)
        self.assertIn("context-block-0000", first)
        self.assertIn("Request 2", first)

    def test_prompt_manifest_preserves_trace_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": {"dataset_revision": "pinned"},
                        "prompts": [
                            {
                                "family": "trajectory-1",
                                "bucket": "8k-16k",
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
        self.assertEqual(prompts[0]["bucket"], "8k-16k")
        self.assertEqual(prompts[0]["source_id"], "session-1")

    def test_prompt_manifest_rejects_missing_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps({"prompts": [{"family": "trajectory-1"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "nonempty prompt"):
                BENCH.read_prompt_manifest(path)

    def test_paired_intervals_are_deterministic(self):
        base = {metric: float(index + 1) for index, metric in enumerate(BENCH.METRICS)}
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
        self.assertAlmostEqual(first["makespan_ms"]["median"], 10.0)


if __name__ == "__main__":
    unittest.main()
