from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
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


def summary(
    output: str,
    ttft: float,
    cache_hits: int = 1,
    requests: int = 1,
) -> dict:
    return {
        "requests": requests,
        "successful": requests,
        "cache_hits": cache_hits,
        "cache_misses": requests - cache_hits,
        "matched_prefix_tokens_median": 100,
        "suffix_prefill_tokens_median": 5,
        "ttft_ms_p50": ttft,
        "ttft_ms_p99": ttft,
        "tpot_ms_p50": 2.0,
        "matched_prefix_tokens": [100] * requests,
        "suffix_prefill_tokens": [5] * requests,
        "ttft_ms": [ttft] * requests,
        "tpot_ms": [2.0] * requests,
        "outputs": [output] * requests,
        "outputs_by_prompt": {"prompt": [output]},
    }


def cell(
    version: str,
    cache: str,
    output: str,
    ttft: float,
    scenario: str = "divergent",
    cache_hits: int = 1,
    concurrency: int = 1,
    requests: int = 1,
    round_number: int = 1,
) -> dict:
    return {
        "version": version,
        "cache": cache,
        "round": round_number,
        "suspect_log_lines": [],
        "observations": [
            {
                "scenario": scenario,
                "concurrency": concurrency,
                "summary": summary(output, ttft, cache_hits, requests),
            }
        ],
    }


class RadixCacheAbTest(unittest.TestCase):
    def test_waits_for_delayed_summary_telemetry(self) -> None:
        class Process:
            @staticmethod
            def poll() -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("")

            def emit() -> None:
                time.sleep(0.02)
                log.write_text(json.dumps({"event": BENCH.SUMMARY_EVENT}) + "\n")

            writer = threading.Thread(target=emit)
            writer.start()
            events = BENCH.wait_for_json_events(
                log,
                BENCH.SUMMARY_EVENT,
                1,
                Process(),
                timeout_seconds=1,
            )
            writer.join()

        self.assertEqual(len(events), 1)

    def test_fails_explicitly_when_summary_telemetry_never_arrives(self) -> None:
        class Process:
            @staticmethod
            def poll() -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("")
            with self.assertRaisesRegex(
                RuntimeError,
                "observed 0/1 events",
            ):
                BENCH.wait_for_json_events(
                    log,
                    BENCH.SUMMARY_EVENT,
                    1,
                    Process(),
                    timeout_seconds=0.01,
                )

    def test_stage_config_offloads_model_layers_explicitly(self) -> None:
        case = BENCH.ModelCase(
            key="test",
            family="test",
            model_id="local/test",
            model_path=Path("/tmp/test.gguf"),
            layer_end=8,
            payload="resident-kv",
        )

        class Harness:
            @staticmethod
            def model_sha256(_path: Path) -> str:
                return "sha256"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.json"
            BENCH.write_config(Harness(), path, case, False, 4096, 2, 999)
            config = json.loads(path.read_text())
        self.assertEqual(config["n_gpu_layers"], 999)

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
            "case": {"payload": "resident-kv"},
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

    def test_recurrent_gate_requires_only_exact_checkpoint_hits(self) -> None:
        cells = []
        for version in ("old", "new"):
            for cache in ("cold", "warm"):
                for scenario in ("exact", "divergent", "coding"):
                    hits = int(cache == "warm" and scenario == "exact")
                    cells.append(
                        cell(
                            version,
                            cache,
                            "correct",
                            25 if cache == "warm" else 100,
                            scenario=scenario,
                            cache_hits=hits,
                        )
                    )
        rows = BENCH.aggregate(cells)
        case_result = {
            "case": {"payload": "kv-recurrent"},
            "cells": cells,
            "aggregate": rows,
            "output_parity": BENCH.parity(rows),
            "cache_output_preservation": BENCH.cache_preservation(rows),
        }
        self.assertEqual(BENCH.evaluate_gate(case_result), {"passed": True, "failures": []})

    def test_concurrent_gate_tolerates_one_transient_miss_per_round(self) -> None:
        cells = []
        for round_number in (1, 2):
            for version, hits in (("old", 4), ("new", 3)):
                for cache in ("cold", "warm"):
                    cells.append(
                        cell(
                            version,
                            cache,
                            "correct",
                            25 if cache == "warm" else 100,
                            cache_hits=hits if cache == "warm" else 0,
                            concurrency=4,
                            requests=4,
                            round_number=round_number,
                        )
                    )
        rows = BENCH.aggregate(cells)
        case_result = {
            "case": {"payload": "resident-kv"},
            "cells": cells,
            "aggregate": rows,
            "output_parity": BENCH.parity(rows),
            "cache_output_preservation": BENCH.cache_preservation(rows),
        }
        self.assertEqual(BENCH.evaluate_gate(case_result), {"passed": True, "failures": []})

    def test_concurrent_gate_rejects_hit_regression_beyond_round_tolerance(self) -> None:
        cells = []
        for round_number in (1, 2):
            for version, hits in (("old", 4), ("new", 2)):
                for cache in ("cold", "warm"):
                    cells.append(
                        cell(
                            version,
                            cache,
                            "correct",
                            25 if cache == "warm" else 100,
                            cache_hits=hits if cache == "warm" else 0,
                            concurrency=4,
                            requests=4,
                            round_number=round_number,
                        )
                    )
        rows = BENCH.aggregate(cells)
        case_result = {
            "case": {"payload": "resident-kv"},
            "cells": cells,
            "aggregate": rows,
            "output_parity": BENCH.parity(rows),
            "cache_output_preservation": BENCH.cache_preservation(rows),
        }
        gate = BENCH.evaluate_gate(case_result)
        self.assertFalse(gate["passed"])
        self.assertTrue(
            any(
                "regressed beyond the 2-request round tolerance" in failure
                for failure in gate["failures"]
            )
        )

    def test_new_n1_output_mismatch_only_fails_when_old_preserves_output(self) -> None:
        def gate_for_old_warm(old_warm_output: str) -> dict:
            cells = [
                cell("old", "cold", "baseline", 100),
                cell("old", "warm", old_warm_output, 25),
                cell("new", "cold", "baseline", 100),
                cell("new", "warm", "changed", 20),
            ]
            rows = BENCH.aggregate(cells)
            return BENCH.evaluate_gate(
                {
                    "case": {"payload": "resident-kv"},
                    "cells": cells,
                    "aggregate": rows,
                    "output_parity": BENCH.parity(rows),
                    "cache_output_preservation": BENCH.cache_preservation(rows),
                }
            )

        self.assertFalse(gate_for_old_warm("baseline")["passed"])
        self.assertTrue(gate_for_old_warm("changed")["passed"])


if __name__ == "__main__":
    unittest.main()
