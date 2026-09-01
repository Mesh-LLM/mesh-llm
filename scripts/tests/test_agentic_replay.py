from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evals/agentic-replay.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agentic_replay", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


class AgenticReplayTest(unittest.TestCase):
    def specs(self):
        return [
            BENCH.RefSpec("rc8", "v0.76.0-rc8", "a" * 40),
            BENCH.RefSpec("main", "origin/main", "b" * 40),
        ]

    def test_ab_order_reverses_every_other_pass(self) -> None:
        order = BENCH.ab_order(self.specs(), 3)

        self.assertEqual(
            [(pass_index, spec.label) for pass_index, spec in order],
            [
                (0, "rc8"),
                (0, "main"),
                (1, "main"),
                (1, "rc8"),
                (2, "rc8"),
                (2, "main"),
            ],
        )

    def test_trajectory_replay_uses_recorded_history_in_strict_turn_order(self) -> None:
        trajectory = {
            "session_id": "session-1",
            "source_dataset": "source",
            "agent_framework": "framework",
            "recorded_model": "recorded-model",
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "recorded answer 1"},
                {"role": "user", "content": "tool observation"},
                {"role": "assistant", "content": "recorded answer 2"},
            ],
        }
        calls = []
        original = BENCH.stream_request

        def fake_stream(
            request_id, messages, tools, metadata, model_id, output_tokens, timeout
        ):
            calls.append((request_id, list(messages), list(tools), dict(metadata)))
            return {**metadata, "request_id": request_id, "error": "fixture"}

        BENCH.stream_request = fake_stream
        try:
            BENCH.replay_trajectory(trajectory, "model", 256, 10)
        finally:
            BENCH.stream_request = original

        self.assertEqual([call[3]["assistant_turn"] for call in calls], [0, 1])
        self.assertEqual(
            [message["content"] for message in calls[0][1]],
            ["rules", "task"],
        )
        self.assertEqual(
            [message["content"] for message in calls[1][1]],
            ["rules", "task", "recorded answer 1", "tool observation"],
        )

    def test_trajectory_replay_honors_warmup_turn_limit(self) -> None:
        trajectory = {
            "session_id": "session-1",
            "source_dataset": "source",
            "agent_framework": "framework",
            "recorded_model": "recorded-model",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "observation"},
                {"role": "assistant", "content": "second"},
            ],
        }
        original = BENCH.stream_request
        BENCH.stream_request = lambda *args, **kwargs: {"request_id": args[0]}
        try:
            results = BENCH.replay_trajectory(
                trajectory, "model", 2048, 10, turn_limit=1
            )
        finally:
            BENCH.stream_request = original

        self.assertEqual([result["request_id"] for result in results], ["session-1:0"])

    def test_trajectory_tools_are_stable_and_schema_shaped(self) -> None:
        trajectory = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls_json": json.dumps(
                        [
                            {"type": "function", "function": {"name": "shell"}},
                            {"type": "function", "function": {"name": "editor"}},
                        ]
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls_json": json.dumps(
                        [{"type": "function", "function": {"name": "shell"}}]
                    ),
                },
            ]
        }

        tools = BENCH.trajectory_tools(trajectory)

        self.assertEqual(
            [tool["function"]["name"] for tool in tools], ["editor", "shell"]
        )
        self.assertTrue(
            all(tool["function"]["parameters"]["additionalProperties"] for tool in tools)
        )

    def test_server_command_keeps_mesh_planning_at_defaults(self) -> None:
        command = BENCH.server_command(Path("/product/mesh-llm"), "model-uri")

        self.assertEqual(
            command,
            [
                "/product/mesh-llm",
                "serve",
                "--model",
                "model-uri",
                "--log-format",
                "json",
            ],
        )
        for option in BENCH.FORBIDDEN_STARTUP_OPTIONS:
            self.assertNotIn(option, command)

    def test_isolated_server_state_preserves_the_model_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hf_home = root / "shared-hf"

            env = BENCH.isolated_server_env(
                root / "runtime-bundle", root / "state", hf_home
            )

            self.assertEqual(env["HF_HOME"], str(hf_home))
            self.assertEqual(env["HOME"], str(root / "state/home"))
            self.assertEqual(
                env["MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"],
                str(root / "runtime-bundle"),
            )

    def test_runtime_evidence_collects_logs_without_copying_identity_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            native_log = state / "runtime/123/logs/skippy-native.log"
            identity = state / "home/.mesh-llm/identity.key"
            native_log.parent.mkdir(parents=True)
            identity.parent.mkdir(parents=True)
            native_log.write_text("runtime evidence", encoding="utf-8")
            identity.write_text("secret", encoding="utf-8")

            BENCH.collect_runtime_logs(state, root / "artifact")

            self.assertEqual(
                (root / "artifact/123/logs/skippy-native.log").read_text(
                    encoding="utf-8"
                ),
                "runtime evidence",
            )
            self.assertFalse((root / "artifact/identity.key").exists())

    def test_default_cli_workload_has_external_concurrency_only(self) -> None:
        args = BENCH.parse_args(
            [
                "plan",
                "--ref",
                "rc8=v0.76.0-rc8",
                "--ref",
                "main=origin/main",
                "--model",
                "model-uri",
                "--trajectories-per-framework",
                "4",
            ]
        )

        self.assertEqual(args.concurrency, [1, 2, 4])
        self.assertEqual(args.passes, 2)
        self.assertEqual(args.trajectories_per_framework, 4)
        self.assertEqual(args.framework, ["swe-agent", "mini-swe-agent", "openhands"])
        self.assertEqual(args.max_output_tokens, 2048)
        self.assertEqual(args.warmup_turns, 14)
        self.assertEqual(args.max_isl, 65536)

    def test_cli_rejects_a_single_wave_concurrency_cohort(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                BENCH.parse_args(
                    [
                        "plan",
                        "--ref",
                        "rc8=v0.76.0-rc8",
                        "--ref",
                        "main=origin/main",
                        "--model",
                        "model-uri",
                        "--trajectories-per-framework",
                        "2",
                    ]
                )

    def test_manifest_contract_is_validated_before_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cohorts": {
                            "warmup": [
                                {
                                    "session_id": "session-1",
                                    "source_dataset": "source",
                                    "agent_framework": "framework",
                                    "recorded_model": "model",
                                    "messages": [{"role": "user", "content": "task"}],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing cohorts"):
                BENCH.load_trajectory_cohorts(manifest, ["warmup", "1"])

    def test_warmup_capacity_is_validated_before_builds(self) -> None:
        trajectories = [
            {
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "action"},
                ]
            }
        ]

        with self.assertRaisesRegex(ValueError, "1 assistant turns; 14 required"):
            BENCH.validate_warmup_capacity(trajectories, 14)

    def test_summarize_requests_computes_stream_windows_and_cache_rate(self) -> None:
        requests = [
            {
                "started": 0.0,
                "first_token_at": 1.0,
                "completed": 3.0,
                "ttft_seconds": 1.0,
                "elapsed_seconds": 3.0,
                "generation_seconds": 2.0,
                "completion_tokens": 10,
                "prompt_tokens": 100,
                "cached_tokens": 50,
                "requested_output_tokens": 10,
            },
            {
                "started": 0.0,
                "first_token_at": 2.0,
                "completed": 4.0,
                "ttft_seconds": 2.0,
                "elapsed_seconds": 4.0,
                "generation_seconds": 2.0,
                "completion_tokens": 10,
                "prompt_tokens": 100,
                "cached_tokens": 100,
                "requested_output_tokens": 10,
            },
        ]

        summary = BENCH.summarize_requests(requests, offered_concurrency=2)

        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_request_ids"], [])
        self.assertEqual(summary["budget_exhausted_requests"], 2)
        self.assertEqual(summary["agent_steps_per_second"], 0.5)
        self.assertEqual(summary["workload_output_tokens_per_second"], 5)
        self.assertEqual(summary["decode_tokens_per_second"], 5)
        self.assertEqual(summary["mean_in_flight"], 1.75)
        self.assertEqual(summary["concurrency_utilization_pct"], 87.5)
        self.assertEqual(summary["ttft_samples"], [1.0, 2.0])
        self.assertEqual(summary["ttft_p50_seconds"], 1.5)
        self.assertEqual(summary["ttft_p95_seconds"], 2.0)
        self.assertEqual(summary["cache_pct"], 75)

    def test_pooled_rows_suppress_deltas_for_different_failed_requests(self) -> None:
        def failed_cell(request_id):
            return {
                "concurrency": 1,
                "trajectories": 3,
                "requests": 5,
                "successful_requests": 4,
                "failed_request_ids": [request_id],
                "budget_exhausted_requests": 0,
                "agent_steps_per_second": 1.0,
                "workload_output_tokens_per_second": 2.0,
                "decode_tokens_per_second": 3.0,
                "ttft_p50_seconds": 4.0,
                "ttft_p95_seconds": 5.0,
                "ttft_samples": [4.0],
                "mean_in_flight": 1.0,
                "concurrency_utilization_pct": 100.0,
                "cache_pct": 50.0,
            }

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "rc8",
                    "ref": "rc8",
                    "commit": "a" * 40,
                    "cells": [failed_cell("session-a:1")],
                },
                {
                    "label": "main",
                    "ref": "main",
                    "commit": "b" * 40,
                    "cells": [failed_cell("session-b:1")],
                },
            ]
        )

        main = next(row for row in rows if row["label"] == "main")
        self.assertFalse(main["delta_comparable"])
        self.assertIsNone(main["decode_tokens_per_second_delta_pct"])

    def test_pooled_rows_compare_each_ref_to_first_ref(self) -> None:
        def cell(concurrency, generation, ttft):
            return {
                "concurrency": concurrency,
                "trajectories": 3,
                "requests": 5,
                "successful_requests": 5,
                "budget_exhausted_requests": 5,
                "agent_steps_per_second": generation,
                "workload_output_tokens_per_second": generation - 1,
                "decode_tokens_per_second": generation + 1,
                "ttft_p50_seconds": ttft,
                "ttft_p95_seconds": ttft + 1,
                "ttft_samples": [ttft, ttft + 1],
                "mean_in_flight": concurrency,
                "concurrency_utilization_pct": 100,
                "cache_pct": 50,
            }

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "rc8",
                    "ref": "v0.76.0-rc8",
                    "commit": "a" * 40,
                    "cells": [cell(2, 20, 4)],
                },
                {
                    "label": "main",
                    "ref": "origin/main",
                    "commit": "b" * 40,
                    "cells": [cell(2, 15, 3)],
                },
            ]
        )

        main = next(row for row in rows if row["label"] == "main")
        self.assertEqual(main["agent_steps_per_second_delta_pct"], -25)
        self.assertEqual(main["ttft_p50_seconds_delta_pct"], -25)
        self.assertAlmostEqual(
            main["decode_tokens_per_second_delta_pct"], -23.8095238095
        )
        self.assertEqual(main["decode_tokens_per_second_min"], 16)

    def test_pooled_rows_weight_decode_and_pool_raw_ttft_samples(self) -> None:
        def cell(tokens, generation_seconds, ttft_samples):
            return {
                "concurrency": 1,
                "trajectories": 3,
                "requests": 2,
                "successful_requests": 2,
                "failed_request_ids": [],
                "completion_tokens": tokens,
                "prompt_tokens": 100,
                "cached_tokens": 50,
                "generation_seconds": generation_seconds,
                "workload_window_seconds": 10.0,
                "budget_exhausted_requests": 0,
                "agent_steps_per_second": 0.2,
                "workload_output_tokens_per_second": tokens / 10.0,
                "decode_tokens_per_second": tokens / generation_seconds,
                "ttft_p50_seconds": BENCH.percentile(ttft_samples, 0.5),
                "ttft_p95_seconds": BENCH.percentile(ttft_samples, 0.95),
                "ttft_samples": ttft_samples,
                "mean_in_flight": 1.0,
                "concurrency_utilization_pct": 100.0,
                "cache_pct": 50.0,
            }

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "rc8",
                    "ref": "rc8",
                    "commit": "a" * 40,
                    "cells": [
                        cell(100, 10.0, [1.0, 2.0]),
                        cell(100, 20.0, [10.0, 20.0]),
                    ],
                }
            ]
        )

        self.assertAlmostEqual(rows[0]["decode_tokens_per_second"], 200 / 30)
        self.assertEqual(rows[0]["ttft_p50_seconds"], 2.0)
        self.assertEqual(rows[0]["ttft_p95_seconds"], 20.0)
        self.assertEqual(rows[0]["ttft_p50_seconds_min"], 1.0)
        self.assertEqual(rows[0]["ttft_p50_seconds_max"], 10.0)

    def test_report_writes_tables_charts_and_inventory(self) -> None:
        cell = {
            "concurrency": 1,
            "trajectories": 3,
            "requests": 5,
            "successful_requests": 5,
            "budget_exhausted_requests": 5,
            "agent_steps_per_second": 20.0,
            "workload_output_tokens_per_second": 18.0,
            "decode_tokens_per_second": 22.0,
            "ttft_p50_seconds": 1.0,
            "ttft_p95_seconds": 1.2,
            "ttft_samples": [1.0, 1.2],
            "mean_in_flight": 1.0,
            "concurrency_utilization_pct": 100.0,
            "cache_pct": 60.0,
        }
        document = {
            "config": {
                "model": "model",
                "concurrency": [1],
                "warmup_turns": 14,
            },
            "inputs": {
                "dataset": {"revision": "c" * 40},
                "manifest_sha256": "d" * 64,
                "cohorts": {
                    "warmup": {
                        "trajectory_count": 3,
                        "assistant_turns": 30,
                        "framework_trajectories": {
                            "swe-agent": 1,
                            "mini-swe-agent": 1,
                            "openhands": 1,
                        },
                        "framework_assistant_turns": {
                            "swe-agent": 10,
                            "mini-swe-agent": 10,
                            "openhands": 10,
                        },
                    },
                    "1": {
                        "trajectory_count": 3,
                        "assistant_turns": 100,
                        "framework_trajectories": {
                            "swe-agent": 1,
                            "mini-swe-agent": 1,
                            "openhands": 1,
                        },
                        "framework_assistant_turns": {
                            "swe-agent": 20,
                            "mini-swe-agent": 30,
                            "openhands": 50,
                        },
                    }
                },
            },
            "builds": [
                {"label": "rc8"},
                {"label": "main"},
            ],
            "order": [
                {"label": "rc8"},
                {"label": "main"},
            ],
            "results": [
                {
                    "label": "rc8",
                    "ref": "v0.76.0-rc8",
                    "commit": "a" * 40,
                    "cells": [cell],
                },
                {
                    "label": "main",
                    "ref": "origin/main",
                    "commit": "b" * 40,
                    "cells": [{**cell, "agent_steps_per_second": 22.0}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            report = BENCH.write_report(artifact, document)

            self.assertTrue(report.is_file())
            self.assertTrue((artifact / "summary/comparison.csv").is_file())
            self.assertTrue(
                (artifact / "summary/charts/decode-throughput.svg").is_file()
            )
            self.assertTrue(
                (artifact / "summary/charts/workload-output-throughput.svg").is_file()
            )
            self.assertTrue((artifact / "summary/charts/ttft-p50.svg").is_file())
            self.assertTrue((artifact / "artifact-sha256.txt").is_file())
            self.assertIn(
                "Mesh chooses context size",
                report.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "swe-agent 1 / 20",
                report.read_text(encoding="utf-8"),
            )
            self.assertIn("Decode tok/s", report.read_text(encoding="utf-8"))
            self.assertIn("Slot use", report.read_text(encoding="utf-8"))

    def test_plan_records_exact_default_server_command(self) -> None:
        args = SimpleNamespace(
            repo=REPO,
            backend="metal",
            model="model-uri",
            passes=2,
            source_dataset=["swe-smith-claude-3-7-sonnet"],
            framework=["swe-agent", "mini-swe-agent", "openhands"],
            trajectories_per_framework=4,
            min_isl=8192,
            max_isl=65536,
            min_turns=5,
            concurrency=[1, 2, 4],
            max_output_tokens=2048,
            warmup_turns=14,
        )

        plan = BENCH.benchmark_plan(args, self.specs())

        self.assertEqual(
            plan["server_command"],
            [
                "<release-binary>",
                "serve",
                "--model",
                "model-uri",
                "--log-format",
                "json",
            ],
        )
        self.assertEqual(
            [item["label"] for item in plan["order"]],
            ["rc8", "main", "main", "rc8"],
        )
        self.assertEqual(plan["selection"]["measured_unique_trajectory_count"], 36)
        self.assertEqual(plan["selection"]["warmup_unique_trajectory_count"], 12)


if __name__ == "__main__":
    unittest.main()
