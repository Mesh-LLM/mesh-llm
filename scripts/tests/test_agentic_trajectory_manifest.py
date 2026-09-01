from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evals/agentic-trajectory-manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agentic_trajectory_manifest", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MANIFEST = load_module()


def row(framework: str, index: int, assistant_turns: int = 2):
    messages = [{"role": "system", "content": "rules"}]
    for turn in range(assistant_turns):
        messages.extend(
            [
                {"role": "user", "content": f"observation-{turn}"},
                {"role": "assistant", "content": f"action-{turn}"},
            ]
        )
    return {
        "session_id": f"{framework}-{index}",
        "source_dataset": f"source-{framework}",
        "agent_framework": framework,
        "recorded_model": "recorded-model",
        "messages_json": json.dumps(messages),
        "n_turns": assistant_turns,
        "max_isl": 10000,
        "total_tokens": 11000,
    }


class AgenticTrajectoryManifestTest(unittest.TestCase):
    def test_balanced_cohorts_are_disjoint_and_keep_whole_trajectories(self) -> None:
        rows = [
            row(framework, index)
            for framework in ("swe-agent", "mini-swe-agent", "openhands")
            for index in range(4)
        ]

        cohorts = MANIFEST.build_cohorts(
            rows,
            ["1", "4"],
            ["swe-agent", "mini-swe-agent", "openhands"],
            2,
        )

        self.assertEqual(len(cohorts["1"]), 6)
        self.assertEqual(len(cohorts["4"]), 6)
        first_ids = {item["session_id"] for item in cohorts["1"]}
        second_ids = {item["session_id"] for item in cohorts["4"]}
        self.assertFalse(first_ids & second_ids)
        for cohort in cohorts.values():
            self.assertEqual(
                {framework: sum(item["agent_framework"] == framework for item in cohort)
                 for framework in ("swe-agent", "mini-swe-agent", "openhands")},
                {"swe-agent": 2, "mini-swe-agent": 2, "openhands": 2},
            )
            self.assertTrue(all(item["assistant_turns"] == 2 for item in cohort))

    def test_manifest_reports_exact_trajectory_and_turn_counts(self) -> None:
        cohorts = {
            "1": [
                {
                    **row("swe-agent", 0, assistant_turns=3),
                    "assistant_turns": 3,
                }
            ]
        }
        document = MANIFEST.manifest_document(cohorts, {"dataset_revision": "abc"})

        self.assertEqual(document["metadata"]["cohorts"]["1"]["trajectory_count"], 1)
        self.assertEqual(document["metadata"]["cohorts"]["1"]["assistant_turns"], 3)
        self.assertEqual(
            document["metadata"]["cohorts"]["1"]["framework_trajectories"],
            {"swe-agent": 1},
        )

    def test_tool_calls_and_ids_are_validated_without_flattening(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls_json": '[{"id":"call-1","type":"function"}]',
                "tool_call_id": None,
            },
            {
                "role": "tool",
                "content": "result",
                "tool_calls_json": None,
                "tool_call_id": "call-1",
            },
        ]

        validated = MANIFEST.validate_messages(json.dumps(messages), "session")

        self.assertEqual(validated, messages)


if __name__ == "__main__":
    unittest.main()
