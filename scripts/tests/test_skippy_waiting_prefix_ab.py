from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO / "evals/skippy-waiting-prefix-ab.py"
    spec = importlib.util.spec_from_file_location("skippy_waiting_prefix_ab", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


def cell(version: str, ttft_ms: float | None) -> dict:
    return {
        "version": version,
        "summary": {
            "requests": 1,
            "successful": int(ttft_ms is not None),
            "cache_hits": 0,
            "suffix_prefill_tokens_total": 0,
            "ttft_ms_p50": ttft_ms,
            "ttft_ms_p95": ttft_ms,
            "makespan_ms": 10,
            "output_tokens_per_second": 0,
            "family_switches": 0,
        },
    }


class WaitingPrefixAbTest(unittest.TestCase):
    def test_reads_a_deterministic_prompt_manifest(self) -> None:
        document = {
            "metadata": {"dataset_revision": "abc123"},
            "prompts": [
                {"family": "one", "prompt": "shared one"},
                {"family": "two", "prompt": "shared two"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(document))
            prompts, metadata = BENCH.read_prompt_manifest(path)

        self.assertEqual(prompts, document["prompts"])
        self.assertEqual(metadata, document["metadata"])

    def test_rejects_an_empty_prompt_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text('{"metadata": {}, "prompts": []}')
            with self.assertRaisesRegex(ValueError, "at least one prompt"):
                BENCH.read_prompt_manifest(path)

    def test_failed_rounds_keep_nullable_percentiles_in_report(self) -> None:
        rows = BENCH.aggregate([cell("old", None), cell("new", 5)])

        self.assertIsNone(rows[0]["ttft_ms_p50_median"])
        self.assertEqual(rows[1]["ttft_ms_p50_median"], 5)
        report = BENCH.report(rows)
        self.assertIn("| TTFT p50 ms | n/a | 5.0 | n/a |", report)


if __name__ == "__main__":
    unittest.main()
