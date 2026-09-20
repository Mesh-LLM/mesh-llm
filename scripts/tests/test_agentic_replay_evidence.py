"""Acceptance tests for complete recorded sessions and actual state restores."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
import agentic_replay_evidence as evidence


def trajectory(session="s"):
    return {
        "session_id": session,
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "recorded one"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": "recorded two"},
        ],
    }


def requests():
    return [
        {
            "session_id": "s",
            "request_id": f"s:{i}",
            "assistant_turn": i,
            "prompt_tokens": 40000 + i * 1000,
            "cached_tokens": i * 39000,
            "ttft_seconds": 1,
        }
        for i in range(2)
    ]


class SessionEvidenceTests(unittest.TestCase):
    def test_exact_order_and_coverage_are_required(self):
        rows = requests()
        self.assertTrue(evidence.complete_sessions([trajectory()], rows)["passed"])
        for invalid in (
            rows[:1],
            rows + rows[:1],
            list(reversed(rows)),
            [{**rows[0], "error": "HTTP 500"}, rows[1]],
        ):
            with self.subTest(rows=invalid):
                self.assertFalse(
                    evidence.complete_sessions([trajectory()], invalid)["passed"]
                )

    def test_long_context_needs_effective_stage_window(self):
        self.assertEqual(
            evidence.runtime_context(
                {"stages": [{"model_id": "m", "ctx_size": 131072}]}, 131072
            ),
            131072,
        )
        for document in ({}, {"stages": [{"model_id": "m", "ctx_size": 32768}]}):
            with self.assertRaises(ValueError):
                evidence.runtime_context(document, 131072)

    def test_prompt_plus_output_must_fit_and_short_sessions_are_not_substituted(self):
        args = ([trajectory()], requests(), 131072, 2048, 32768, lambda m, n: n)
        self.assertTrue(evidence.context_eligibility(*args)["passed"])
        too_long = requests()
        too_long[1]["prompt_tokens"] = 130000
        self.assertFalse(
            evidence.context_eligibility(
                [trajectory()], too_long, 131072, 2048, 32768, lambda m, n: n
            )["passed"]
        )
        self.assertFalse(
            evidence.context_eligibility(
                [trajectory()], requests(), 131072, 2048, 65536, lambda m, n: n
            )["passed"]
        )

    def test_capture_counters_cannot_substitute_for_recurrent_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.log"
            events = [
                self.event(0, "miss"),
                self.event(1, "exact_hit", "kv-recurrent", 39000),
            ]
            path.write_text("\n".join(json.dumps(e) for e in events))
            result = evidence.recurrent_evidence([path], requests())
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["restores"], 1)
            for replacement in (
                self.event(1, "exact_hit", "full-state", 39000),
                self.event(1, "miss"),
                self.event(1, "exact_hit", "kv-recurrent", 999999),
            ):
                path.write_text(
                    "\n".join(json.dumps(e) for e in [events[0], replacement])
                )
                self.assertFalse(
                    evidence.recurrent_evidence([path], requests())["passed"]
                )
            path.write_text(json.dumps(events[1]))
            self.assertFalse(evidence.recurrent_evidence([path], requests())["passed"])

    def test_trivial_restore_does_not_pass_long_context_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.log"
            path.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in (
                        self.event(0, "miss"),
                        self.event(1, "exact_hit", "kv-recurrent", 1),
                    )
                )
            )
            self.assertFalse(
                evidence.recurrent_evidence(
                    [path], requests(), minimum_restored_tokens=32768
                )["passed"]
            )

    @staticmethod
    def event(index, decision, payload=None, tokens=0):
        return {
            "event": "stage.openai_kv_lookup_decision",
            "start_time_unix_nanos": index,
            "attributes": {
                "openai.prompt_cache_key": "s",
                "skippy.kv.decision": decision,
                "skippy.exact_cache.payload_kind": payload,
                "skippy.exact_cache.restored_tokens": tokens,
            },
        }

    def test_same_matrix_command_is_used_for_original_and_repair(self):
        path = ROOT / "scripts/agentic-replay-params.py"
        spec = importlib.util.spec_from_file_location("replay_params_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        matrix = ROOT / "ci/agentic-replay-nightly/matrix.json"
        for model in json.loads(matrix.read_text())["models"]:
            command = module.replay_command(
                matrix, model["family"], ["fixed=HEAD", "base=abc"], "data", "out"
            )
            self.assertIn("--sessions-per-concurrency", command)
            self.assertIn("--minimum-context-tokens", command)
            self.assertEqual(command[command.index("--replay-mode") + 1], "all")
            self.assertEqual(
                "--require-recurrent-restores" in command,
                model["class"] == "hybrid-recurrent",
            )
            self.assertIn("base=abc", command)
