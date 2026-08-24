from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO / "evals/skippy-radix-cache-ab.py"
    spec = importlib.util.spec_from_file_location("skippy_radix_cache_ab", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


def summary(output: str, ttft: float) -> dict:
    return {
        "requests": 1,
        "successful": 1,
        "cache_hits": 1,
        "cache_misses": 0,
        "matched_prefix_tokens_median": 100,
        "suffix_prefill_tokens_median": 5,
        "ttft_ms_p50": ttft,
        "ttft_ms_p99": ttft,
        "tpot_ms_p50": 2.0,
        "matched_prefix_tokens": [100],
        "suffix_prefill_tokens": [5],
        "ttft_ms": [ttft],
        "tpot_ms": [2.0],
        "outputs": [output],
        "outputs_by_prompt": {"prompt": [output]},
    }


def cell(version: str, cache: str, output: str, ttft: float) -> dict:
    return {
        "version": version,
        "cache": cache,
        "suspect_log_lines": [],
        "observations": [
            {
                "scenario": "divergent",
                "concurrency": 1,
                "summary": summary(output, ttft),
            }
        ],
    }


class RadixCacheAbTest(unittest.TestCase):
    def test_divergent_prompts_are_unique_and_nonempty(self) -> None:
        first = BENCH.divergent_prompt("stable", 1, 0)
        second = BENCH.divergent_prompt("stable", 1, 1)
        self.assertTrue(first.startswith("stable"))
        self.assertNotEqual(first, second)

    def test_coding_agent_trace_grows_without_changing_its_prefix(self) -> None:
        first = BENCH.coding_agent_prompt("stable", 1, 0)
        second = BENCH.coding_agent_prompt("stable", 1, 1)
        self.assertTrue(second.startswith(first.removesuffix("Assistant: return the latest invariant only.")))
        self.assertGreater(len(second), len(first))

    def test_empty_telemetry_does_not_invent_cache_metrics(self) -> None:
        result = BENCH.summarize_requests(
            [
                {
                    "content": "ok",
                    "first_content": "ok",
                    "elapsed_ms": 10,
                    "ttft_ms": 4,
                    "tpot_ms": 2,
                    "prompt_sha256": "prompt",
                }
            ],
            [],
        )
        self.assertIsNone(result["matched_prefix_tokens_median"])
        self.assertIsNone(result["suffix_prefill_tokens_median"])
        self.assertEqual(result["outputs_by_prompt"], {"prompt": ["ok"]})

    def test_cache_lift_and_per_prompt_preservation_are_separate(self) -> None:
        rows = BENCH.aggregate(
            [
                cell("old", "cold", "correct", 100),
                cell("old", "warm", "stale", 30),
                cell("new", "cold", "correct", 100),
                cell("new", "warm", "correct", 20),
            ]
        )
        warm = {
            (row["version"], row["cache"]): row
            for row in rows
            if row["cache"] == "warm"
        }
        self.assertEqual(warm[("old", "warm")]["cache_lift_ttft_ms"], 70)
        self.assertEqual(warm[("new", "warm")]["cache_lift_ttft_ms"], 80)
        preservation = {
            row["version"]: row["cache_preserves_output"]
            for row in BENCH.cache_preservation(rows)
        }
        self.assertEqual(preservation, {"old": False, "new": True})
        case_result = {
            "cells": [
                cell("old", "cold", "correct", 100),
                cell("old", "warm", "stale", 30),
                cell("new", "cold", "correct", 100),
                cell("new", "warm", "correct", 20),
            ],
            "aggregate": rows,
            "output_parity": BENCH.parity(rows),
            "cache_output_preservation": BENCH.cache_preservation(rows),
        }
        self.assertEqual(BENCH.evaluate_gate(case_result), {"passed": True, "failures": []})


if __name__ == "__main__":
    unittest.main()
