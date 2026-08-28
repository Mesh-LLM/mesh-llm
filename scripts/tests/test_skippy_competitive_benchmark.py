from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evals/skippy-competitive-benchmark.py"
CONFIG = REPO / "evals/skippy-competitive-benchmark.json"


def load_module():
    spec = importlib.util.spec_from_file_location("skippy_competitive_benchmark", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


class CompetitiveBenchmarkTest(unittest.TestCase):
    def test_checked_in_plan_covers_both_platforms_all_models_and_full_ladder(self) -> None:
        config = BENCH.load_config(CONFIG)
        plan = BENCH.build_plan(
            config,
            ["cuda", "metal"],
            config["models"],
            ["synthetic", "thoughtworks"],
        )

        self.assertEqual(plan["cell_count"], 432)
        self.assertEqual(
            sorted({cell["concurrency"] for cell in plan["cells"]}),
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
        self.assertEqual(
            sorted({cell["platform"] for cell in plan["cells"]}),
            ["cuda", "metal"],
        )
        self.assertEqual(len({cell["model"] for cell in plan["cells"]}), 3)
        trace = [cell for cell in plan["cells"] if cell["workload"] == "thoughtworks"]
        self.assertTrue(all(cell["prompt_count"] % cell["concurrency"] == 0 for cell in trace))
        self.assertTrue(all(cell["prompt_count"] >= cell["concurrency"] for cell in trace))

    def test_trace_arm_order_alternates_to_reduce_time_order_bias(self) -> None:
        config = BENCH.load_config(CONFIG)
        plan = BENCH.build_plan(
            config,
            ["metal"],
            [config["models"][0]],
            ["thoughtworks"],
        )
        cells = plan["cells"]

        self.assertEqual([cell["arm"] for cell in cells[:4]], ["llama", "mesh", "mesh", "llama"])

    def test_manifest_verification_rejects_provenance_drift(self) -> None:
        config = BENCH.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "prompts.json"
            document = {
                "metadata": {"rows": [{"session_id": "wrong"}]},
                "prompts": [
                    {"family": "fixture", "prompt": "trace"}
                    for _ in range(256)
                ],
            }
            manifest.write_text(json.dumps(document), encoding="utf-8")
            config["thoughtworks"]["selection"]["manifest_sha256"] = BENCH.sha256(manifest)

            with self.assertRaisesRegex(RuntimeError, "row provenance"):
                BENCH.verify_manifest(manifest, config)

    def test_server_commands_keep_raw_and_mesh_lane_counts_equal(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][0]
        args = SimpleNamespace(mesh_binary=Path("mesh"), llama_binary=Path("llama"))
        common = (model, Path("model.gguf"), Path("stage.json"), 19000, 16384, 2, 8)

        mesh = BENCH.server_command("mesh", args, *common, True)
        raw = BENCH.server_command("llama", args, *common, True)

        self.assertEqual(mesh[mesh.index("--generation-concurrency") + 1], "2")
        self.assertEqual(raw[raw.index("--parallel") + 1], "2")
        self.assertIn("--kv-unified", raw)
        self.assertNotIn("--no-cache-prompt", raw)

    def test_report_writes_csv_svg_markdown_and_hash_manifest(self) -> None:
        config = BENCH.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, throughput in (("llama", 100.0), ("mesh", 120.0)):
                arm_dir = root / "data" / "metal" / "llama32-dense" / arm
                arm_dir.mkdir(parents=True)
                (arm_dir / "tg-8-c-1.json").write_text(
                    json.dumps(
                        {
                            "benchmarks": [
                                {
                                    "tg_throughput": {"mean": throughput},
                                    "e2e_ttft": {"mean": 10.0},
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                (arm_dir / "tg-8-c-1-progress.jsonl").write_text(
                    json.dumps({"type": "request_end", "error": None}) + "\n",
                    encoding="utf-8",
                )
                (arm_dir / "tg-8-c-1.out").write_text("", encoding="utf-8")
                (arm_dir / "status.tsv").write_text(
                    "tg\tconcurrency\texit_code\n8\t1\t0\n", encoding="utf-8"
                )
                (arm_dir / "parity.json").write_text(
                    json.dumps(
                        {
                            "cells": [
                                {
                                    "concurrency": 1,
                                    "results": [
                                        {
                                            "request_index": 0,
                                            "status": 200,
                                            "content_sha256": "same",
                                        }
                                    ],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                trace_dir = root / "trace" / "metal" / "llama32-dense" / "c-1" / arm
                trace_dir.mkdir(parents=True)
                (trace_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "platform": "metal",
                            "model": "llama32-dense",
                            "arm": arm,
                            "concurrency": 1,
                            "successful_requests": 60,
                            "failed_requests": 0,
                            "output_tokens_per_second": throughput,
                        }
                    ),
                    encoding="utf-8",
                )

            BENCH.report(argparse.Namespace(artifact=root), config)

            report = (root / "summary" / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Correctness gate: **PASS**", report)
            self.assertIn("+20.00%", report)
            self.assertTrue((root / "summary" / "synthetic.csv").is_file())
            self.assertTrue((root / "summary" / "thoughtworks.csv").is_file())
            self.assertTrue(list((root / "summary" / "charts").glob("*.svg")))
            self.assertTrue((root / "artifact-sha256.txt").is_file())


if __name__ == "__main__":
    unittest.main()
