"""Structural contract tests for the process-level logging recovery harness."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "qa-logging-recovery.sh"


class LoggingRecoveryHarnessTests(unittest.TestCase):
    def run_plan(self, *args: str) -> dict[str, object]:
        completed = subprocess.run(
            [str(SCRIPT), "--current-binary", "/not/an/executable", *args, "--print-plan"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], cwd=ROOT, check=True)

    def test_print_plan_is_side_effect_free_and_declares_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "must-not-exist"
            plan = self.run_plan("--evidence-dir", str(evidence))
            self.assertFalse(evidence.exists(), "--print-plan must not create evidence")

        self.assertEqual(plan["script"], "qa-logging-recovery.sh")
        self.assertEqual(
            plan["checks"],
            [
                "logging_restart_privacy",
                "logging_retention_cascade",
                "logging_trusted_local_rejection",
                "logging_sse_recovery",
                "logging_fail_open",
                "logging_fail_open_inference",
                "cleanup",
            ],
        )
        self.assertEqual(
            plan["evidence_files"][:5],
            ["manifest.json", "commands.jsonl", "results.jsonl", "summary.json", "summary.md"],
        )

    def test_optional_deterministic_plugin_is_explicit_prerequisite(self) -> None:
        without_plugin = self.run_plan()
        self.assertFalse(without_plugin["deterministic_openai_endpoint_supplied"])
        self.assertEqual(
            without_plugin["optional_plugin_behavior"]["without_endpoint"],
            {"logging_fail_open_inference": "PREREQ"},
        )

        with_plugin = self.run_plan(
            "--deterministic-openai-endpoint",
            "http://127.0.0.1:18080/v1",
            "--deterministic-openai-model",
            "deterministic-test-model",
        )
        self.assertTrue(with_plugin["deterministic_openai_endpoint_supplied"])
        self.assertEqual(with_plugin["deterministic_openai_model"], "deterministic-test-model")
        self.assertEqual(
            with_plugin["optional_plugin_behavior"]["with_endpoint"],
            {"logging_fail_open_inference": "execute"},
        )

    def test_uses_v6_delete_contract_and_private_marker_assertions(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("/api/logs/requests/$restart_request_id/delete", source)
        self.assertIn('"operationId"', source)
        self.assertIn("assert_no_private_marker", source)
        self.assertIn("typed_forbidden_response", source)
        self.assertIn("Last-Event-ID: v1:0.0.0", source)
        self.assertIn("event: replay_gap", source)
        self.assertIn("/api/logs/events?channel=requests&cursor=v1%3A0.0.0", source)
        self.assertIn("list-before-restart.json", source)
        self.assertIn("detail-after-restart.json", source)
        self.assertNotIn("Authorization:", source)
        self.assertNotIn("Bearer ", source)
        self.assertNotIn("sk-", source)
        self.assertNotIn("v5", source.lower())
        self.assertNotIn("websocket", source.lower())

    def test_wait_status_binds_label_before_constructing_output_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'local label="$1" console_port="$2" second\n'
            '    local output="$STATUS_DIR/$label.json"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
