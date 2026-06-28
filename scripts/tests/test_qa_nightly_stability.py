from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "qa-nightly-stability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qa_nightly_stability", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NightlyStabilityHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = load_module()

    def test_parses_models_and_agent_smokes(self) -> None:
        self.assertEqual(self.harness.parse_csv("auto, mesh"), ["auto", "mesh"])
        self.assertEqual(self.harness.parse_csv(""), [])
        self.assertEqual(self.harness.parse_agent_smokes("opencode, goose"), ["opencode", "goose"])

        with self.assertRaisesRegex(ValueError, "unknown agent smoke"):
            self.harness.parse_agent_smokes("opencode,unknown")

    def test_build_plan_is_side_effect_free_and_repeatable(self) -> None:
        plan = self.harness.build_plan(
            base_url="http://127.0.0.1:9337",
            models=["auto", "mesh"],
            attempts=2,
            output_dir=Path("target/nightly-stability/example"),
            agent_smokes=["opencode", "pi", "goose"],
            skip_streaming=False,
            timeout=120.0,
            mesh_binary=None,
            release_attestation_expected_status=None,
        )

        self.assertEqual(plan["name"], "nightly-stability")
        self.assertEqual(plan["endpoint"], "http://127.0.0.1:9337/v1")
        self.assertEqual(plan["models"], ["auto", "mesh"])
        self.assertEqual(plan["attempts"], 2)
        self.assertEqual([step["name"] for step in plan["steps"]], [
            "openai-surface-probe",
            "tool-call-reliability",
            "opencode-agent-smoke",
            "pi-agent-smoke",
            "goose-agent-smoke",
        ])
        self.assertNotIn("--skip-streaming", json.dumps(plan), "streaming should not be skipped")

    def test_build_plan_can_skip_streaming(self) -> None:
        plan = self.harness.build_plan(
            base_url="http://127.0.0.1:9337/v1",
            models=["auto"],
            attempts=1,
            output_dir=Path("target/nightly-stability/example"),
            agent_smokes=[],
            skip_streaming=True,
            timeout=30.0,
            mesh_binary=None,
            release_attestation_expected_status=None,
        )
        command = plan["steps"][1]["command"]
        self.assertIn("--skip-streaming", command)

    def test_command_specs_keep_logs_relative_to_output_dir(self) -> None:
        output_dir = Path("nightly-artifacts/stability/12345")
        specs = self.harness.build_command_specs(
            base_url="http://127.0.0.1:9337",
            models=["auto"],
            attempts=1,
            output_dir=output_dir,
            agent_smokes=["opencode"],
            skip_streaming=False,
            timeout=30.0,
        )

        self.assertEqual(
            [spec.log for spec in specs],
            [
                "logs/tool-call-reliability.log",
                "logs/opencode-agent-smoke.log",
            ],
        )

    def test_summarizes_command_results(self) -> None:
        rows = [
            self.harness.CommandResult(
                name="tool-call-reliability",
                status="PASS",
                exit_code=0,
                elapsed_ms=25,
                log="logs/tool.log",
            ),
            self.harness.CommandResult(
                name="opencode-agent-smoke",
                status="FAIL",
                exit_code=1,
                elapsed_ms=50,
                log="logs/opencode.log",
            ),
        ]
        summary = self.harness.summarize_results(rows)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total"], 2)

    def test_missing_optional_agent_cli_records_prereq(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = self.harness.run_command(
                self.harness.CommandSpec(
                    name="missing-agent-smoke",
                    command=["/definitely/missing"],
                    env={},
                    log=str(Path("logs") / "missing-agent-smoke.log"),
                    prerequisite="mesh-llm-definitely-missing-agent-cli",
                ),
                out,
            )
            log = out / "logs" / "missing-agent-smoke.log"
            self.assertTrue(log.exists())

        self.assertEqual(result.status, "PREREQ")
        self.assertEqual(result.exit_code, 0)

    def test_summarizes_probe_results(self) -> None:
        rows = [
            self.harness.ProbeResult(
                model="auto",
                attempt=1,
                phase="chat",
                ok=True,
                detail="matched sentinel",
                elapsed_ms=20,
                status_code=200,
                ttft_ms=None,
            ),
            self.harness.ProbeResult(
                model="mesh",
                attempt=1,
                phase="stream_chat",
                ok=False,
                detail="stream returned no choices",
                elapsed_ms=40,
                status_code=None,
                ttft_ms=None,
            ),
        ]
        summary = self.harness.summarize_probe_results(rows)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total"], 2)

    def test_chat_probe_reports_http_failure_without_crashing(self) -> None:
        with mock.patch.object(
            self.harness,
            "post_json",
            side_effect=RuntimeError("HTTP 403: error code: 1010"),
        ):
            result = self.harness.run_chat_probe("https://meshllm.cloud/v1", "auto", 1, 1.0)

        self.assertFalse(result.ok)
        self.assertEqual(result.model, "auto")
        self.assertEqual(result.attempt, 1)
        self.assertEqual(result.phase, "chat")
        self.assertIn("HTTP 403", result.detail)
        self.assertIsNone(result.actual_model)
        self.assertIsNone(result.tok_per_sec)

    def test_chat_probe_reports_validation_failure_with_response_metadata(self) -> None:
        response = {
            "model": "auto-resolved",
            "usage": {"completion_tokens": 8},
            "choices": [{"message": {"content": "WRONG"}}],
        }
        with mock.patch.object(self.harness, "post_json", return_value=(response, 202)):
            result = self.harness.run_chat_probe("https://meshllm.cloud/v1", "auto", 1, 1.0)

        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 202)
        self.assertIsNone(result.ttft_ms)
        self.assertEqual(result.actual_model, "auto-resolved")
        self.assertIsNotNone(result.tok_per_sec)
        self.assertIn("expected exactly STABILITY_OK", result.detail)

    def test_stream_chat_probe_reports_http_failure_without_crashing(self) -> None:
        with mock.patch.object(
            self.harness,
            "post_json_stream",
            side_effect=RuntimeError("HTTP 403: error code: 1010"),
        ):
            result = self.harness.run_stream_chat_probe("https://meshllm.cloud/v1", "mesh", 1, 1.0)

        self.assertFalse(result.ok)
        self.assertEqual(result.model, "mesh")
        self.assertEqual(result.attempt, 1)
        self.assertEqual(result.phase, "stream_chat")
        self.assertIn("HTTP 403", result.detail)
        self.assertIsNone(result.actual_model)
        self.assertIsNone(result.tok_per_sec)

    def test_stream_chat_probe_reports_validation_failure_with_response_metadata(self) -> None:
        chunks = [
            {"model": "mesh-resolved", "choices": [{"delta": {"content": "WRONG"}}]},
            {"usage": {"completion_tokens": 7}},
        ]
        with mock.patch.object(self.harness, "post_json_stream", return_value=(chunks, 206, 123)):
            result = self.harness.run_stream_chat_probe("https://meshllm.cloud/v1", "mesh", 1, 1.0)

        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 206)
        self.assertEqual(result.ttft_ms, 123)
        self.assertEqual(result.actual_model, "mesh-resolved")
        self.assertIsNotNone(result.tok_per_sec)
        self.assertIn("expected exactly STREAM_OK", result.detail)

    def test_write_evidence_outputs_machine_readable_files(self) -> None:
        rows = [
            self.harness.CommandResult(
                name="tool-call-reliability",
                status="PASS",
                exit_code=0,
                elapsed_ms=25,
                log="logs/tool.log",
            )
        ]
        plan = self.harness.build_plan(
            base_url="http://127.0.0.1:9337",
            models=["auto"],
            attempts=1,
            output_dir=Path("target/nightly-stability/example"),
            agent_smokes=[],
            skip_streaming=False,
            timeout=30.0,
            mesh_binary=None,
            release_attestation_expected_status=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            probe_rows = [
                self.harness.ProbeResult(
                    model=None,
                    attempt=None,
                    phase="models",
                    ok=True,
                    detail="1 models",
                    elapsed_ms=10,
                    status_code=200,
                    ttft_ms=None,
                )
            ]
            self.harness.write_evidence(out, plan, rows, probe_rows)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            commands = [
                json.loads(line)
                for line in (out / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            results = [
                json.loads(line)
                for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(manifest["name"], "nightly-stability")
        self.assertTrue(summary["ok"])
        self.assertEqual(commands[0]["name"], "tool-call-reliability")
        self.assertEqual(results[0]["phase"], "models")

    def test_summary_markdown_includes_timing_snapshot(self) -> None:
        summary = {
            "ok": True,
            "total": 2,
            "passed": 2,
            "failed": 0,
            "prereq": 0,
            "elapsed_ms": 35,
            "commands": {
                "passed": 1,
                "failed": 0,
                "prereq": 0,
                "elapsed_ms": 25,
            },
            "probes": {
                "passed": 1,
                "failed": 0,
                "prereq": 0,
                "elapsed_ms": 10,
            },
        }
        rendered = self.harness.render_summary_markdown(summary, [], [], None)

        self.assertIn("## Timing Snapshot", rendered)
        self.assertIn("| OpenAI surface probes | 1 | 0 | 0 | 10 |", rendered)
        self.assertIn("| Command probes | 1 | 0 | 0 | 25 |", rendered)


if __name__ == "__main__":
    unittest.main()
