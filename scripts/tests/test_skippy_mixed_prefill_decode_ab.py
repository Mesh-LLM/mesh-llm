#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
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

    def test_stage_config_connects_two_stage_topology(self):
        args = Namespace(
            split_layer=14,
            model_id="test-model",
            model=Path("/tmp/model.gguf"),
            model_sha256="abc123",
            ctx_size=32768,
            lanes=12,
            n_batch=1024,
            n_ubatch=256,
            n_gpu_layers=999,
        )

        stage0 = BENCH.stage_config(args, 0, 0, 14, 9000, 9001)
        stage1 = BENCH.stage_config(args, 1, 14, 27, 9001, 9000)

        self.assertEqual(stage0["downstream"]["endpoint"], "tcp://127.0.0.1:9001")
        self.assertIsNone(stage0["upstream"])
        self.assertEqual(stage1["upstream"]["endpoint"], "tcp://127.0.0.1:9000")
        self.assertIsNone(stage1["downstream"])
        self.assertEqual(stage0["layer_end"], stage1["layer_start"])

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

    def test_scheduler_events_accept_feature_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-0.log"
            path.write_text(
                json.dumps(
                    {
                        "event": "stage.scheduler_feature_iteration",
                        "attributes": {"skippy.scheduler.token_count": 7},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            events = BENCH.scheduler_events(path)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["skippy.scheduler.token_count"], 7)
        self.assertEqual(events[0]["_event"], "stage.scheduler_feature_iteration")

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
