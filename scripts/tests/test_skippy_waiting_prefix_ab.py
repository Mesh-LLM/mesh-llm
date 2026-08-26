from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
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
    def test_warm_profile_accepts_the_measured_neutral_boundary(self) -> None:
        catalog = json.loads((REPO / "evals/skippy-scheduler-fixtures.json").read_text())
        profile = catalog["profiles"]["warm-affinity"]
        rows = [
            {
                "version": "old",
                "successful": 24,
                "suffix_prefill_tokens_median": 8237.0,
                "family_switches_median": 1.0,
                "ttft_ms_p50_median": 3161.1,
                "ttft_ms_p95_median": 3313.7,
                "makespan_ms_median": 5326.8,
                "output_tokens_per_second_median": 71.34,
            },
            {
                "version": "new",
                "successful": 24,
                "suffix_prefill_tokens_median": 8237.0,
                "family_switches_median": 1.0,
                "ttft_ms_p50_median": 3190.0,
                "ttft_ms_p95_median": 3305.4,
                "makespan_ms_median": 5331.8,
                "output_tokens_per_second_median": 71.27,
            },
        ]

        acceptance = BENCH.evaluate_acceptance(rows, profile)

        self.assertTrue(acceptance["passed"])

    def test_zero_baseline_regression_fails_closed(self) -> None:
        catalog = json.loads((REPO / "evals/skippy-scheduler-fixtures.json").read_text())
        profile = catalog["profiles"]["warm-affinity"]
        rows = [
            {
                "version": "old",
                "successful": 24,
                "suffix_prefill_tokens_median": 0.0,
                "family_switches_median": 0.0,
                "ttft_ms_p50_median": 10.0,
                "ttft_ms_p95_median": 10.0,
                "makespan_ms_median": 10.0,
                "output_tokens_per_second_median": 10.0,
            },
            {
                "version": "new",
                "successful": 24,
                "suffix_prefill_tokens_median": 1.0,
                "family_switches_median": 1.0,
                "ttft_ms_p50_median": 10.0,
                "ttft_ms_p95_median": 10.0,
                "makespan_ms_median": 10.0,
                "output_tokens_per_second_median": 10.0,
            },
        ]

        acceptance = BENCH.evaluate_acceptance(rows, profile)

        self.assertFalse(acceptance["passed"])
        self.assertIsNone(BENCH.delta(0.0, 1.0))

    def test_eviction_profile_enforces_hardware_acceptance(self) -> None:
        catalog = json.loads((REPO / "evals/skippy-scheduler-fixtures.json").read_text())
        profile = catalog["profiles"]["agentic-eviction-pressure"]
        rows = [
            {
                "version": "old",
                "successful": 64,
                "suffix_prefill_tokens_median": 92061.5,
                "family_switches_median": 14.0,
                "ttft_ms_p95_median": 38646.5,
                "makespan_ms_median": 41132.9,
                "output_tokens_per_second_median": 12.02,
            },
            {
                "version": "new",
                "successful": 64,
                "suffix_prefill_tokens_median": 66735.5,
                "family_switches_median": 10.0,
                "ttft_ms_p95_median": 27901.1,
                "makespan_ms_median": 30328.4,
                "output_tokens_per_second_median": 16.26,
            },
        ]

        acceptance = BENCH.evaluate_acceptance(rows, profile)

        self.assertTrue(acceptance["passed"])
        self.assertTrue(all(check["passed"] for check in acceptance["checks"]))
        rows[1]["output_tokens_per_second_median"] = rows[0][
            "output_tokens_per_second_median"
        ]
        self.assertFalse(BENCH.evaluate_acceptance(rows, profile)["passed"])

    def test_checked_in_fixture_profile_owns_the_workload_shape(self) -> None:
        args = Namespace(
            fixture_profile="agentic-eviction-pressure",
            fixture_catalog=REPO / "evals/skippy-scheduler-fixtures.json",
            rounds=1,
            families=1,
        )

        selected, catalog_hash = BENCH.apply_fixture_profile(args)

        self.assertEqual(args.rounds, 4)
        self.assertEqual(args.families, 8)
        self.assertEqual(args.admission_concurrency, 16)
        self.assertEqual(selected["corpus"]["kind"], "hf")
        self.assertEqual(len(catalog_hash), 64)

    def test_fixture_input_validation_pins_model_and_manifest_mode(self) -> None:
        catalog = json.loads((REPO / "evals/skippy-scheduler-fixtures.json").read_text())
        warm = catalog["profiles"]["warm-affinity"]
        agentic = catalog["profiles"]["agentic-eviction-pressure"]
        model = warm["model"]

        BENCH.validate_fixture_inputs(warm, model["id"], model["sha256"], None)
        with self.assertRaisesRegex(ValueError, "model id"):
            BENCH.validate_fixture_inputs(warm, "different", model["sha256"], None)
        with self.assertRaisesRegex(ValueError, "do not accept"):
            BENCH.validate_fixture_inputs(warm, model["id"], model["sha256"], Path("x"))
        with self.assertRaisesRegex(ValueError, "require a prepared"):
            BENCH.validate_fixture_inputs(agentic, model["id"], model["sha256"], None)

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
