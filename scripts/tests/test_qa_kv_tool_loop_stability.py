import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "qa-kv-tool-loop-stability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qa_kv_tool_loop_stability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KvToolLoopStabilityTests(unittest.TestCase):
    def test_plan_declares_tool_loop_cache_and_log_scan(self):
        harness = load_module()
        plan = harness.build_plan(
            base_url="http://localhost:9337",
            models=["Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m"],
            attempts=3,
            pressure_turns=8,
            timeout=90.0,
            output_dir=Path("target/kv-tool-loop-stability/latest"),
            min_cached_tokens=2048,
            suffix_prefill_limit=256,
            native_logs=[Path("skippy-native.log")],
        )

        self.assertEqual(plan["name"], "kv-tool-loop-stability")
        self.assertEqual(plan["base_url"], "http://localhost:9337/v1")
        self.assertEqual(plan["attempts"], 3)
        self.assertEqual(plan["pressure_turns"], 8)
        self.assertEqual(plan["timeout_seconds"], 90.0)
        self.assertEqual(plan["min_cached_tokens"], 2048)
        self.assertEqual(plan["suffix_prefill_limit"], 256)
        self.assertEqual(plan["native_log_scan_mode"], "appended_since_run_start")
        self.assertIn("manifest.json", plan["evidence_files"])
        self.assertIn("results.jsonl", plan["evidence_files"])
        self.assertIn("summary.md", plan["evidence_files"])
        self.assertEqual(
            [check["phase"] for check in plan["checks"]],
            ["tool_loop", "same_prefix_cache", "exact_prefix_cache", "native_log_scan"],
        )
        self.assertEqual(plan["checks"][0]["pressure_turns"], 8)
        self.assertEqual(plan["checks"][1]["suffix_prefill_limit"], 256)
        self.assertEqual(plan["checks"][2]["timeout_seconds"], 90.0)
        self.assertEqual(plan["checks"][3]["scan_mode"], "appended_since_run_start")

    def test_parse_native_logs_dedupes_env_and_cli_paths(self):
        harness = load_module()
        original = os.environ.get("MESH_KV_TOOL_LOOP_NATIVE_LOGS")
        os.environ["MESH_KV_TOOL_LOOP_NATIVE_LOGS"] = "skippy-native.log, other.log"
        try:
            logs = harness.parse_native_logs(["skippy-native.log", "third.log"])
        finally:
            if original is None:
                os.environ.pop("MESH_KV_TOOL_LOOP_NATIVE_LOGS", None)
            else:
                os.environ["MESH_KV_TOOL_LOOP_NATIVE_LOGS"] = original

        self.assertEqual(
            [str(path) for path in logs],
            ["skippy-native.log", "other.log", "third.log"],
        )

    def test_tool_loop_requests_keep_stable_prefix_and_vary_tail(self):
        harness = load_module()
        first = harness.build_tool_call_request("direct-model", attempt=1)
        second = harness.build_tool_call_request("direct-model", attempt=2)

        self.assertEqual(first["model"], "direct-model")
        self.assertEqual(first["messages"][0]["content"], second["messages"][0]["content"])
        self.assertNotEqual(first["messages"][1]["content"], second["messages"][1]["content"])
        self.assertEqual(
            first["tool_choice"],
            {"type": "function", "function": {"name": "lookup_probe_fact"}},
        )
        self.assertFalse(first["parallel_tool_calls"])
        self.assertEqual(first["tools"][0]["function"]["name"], "lookup_probe_fact")

    def test_cache_metrics_extract_openai_usage_tokens(self):
        harness = load_module()
        metrics = harness.extract_cache_metrics(
            {
                "usage": {
                    "prompt_tokens": 2240,
                    "prompt_tokens_details": {"cached_tokens": 2176},
                }
            }
        )
        self.assertEqual(metrics.prompt_tokens, 2240)
        self.assertEqual(metrics.cached_tokens, 2176)

        missing = harness.extract_cache_metrics({"usage": {"prompt_tokens": 12}})
        self.assertEqual(missing.prompt_tokens, 12)
        self.assertEqual(missing.cached_tokens, 0)

    def test_cache_threshold_reports_shortfall(self):
        harness = load_module()

        ok, detail = harness.evaluate_cache_threshold(
            harness.CacheMetrics(prompt_tokens=2240, cached_tokens=2176),
            min_cached_tokens=2048,
            suffix_prefill_limit=256,
        )
        self.assertTrue(ok)
        self.assertIn("cached_tokens=2176", detail)

        ok, detail = harness.evaluate_cache_threshold(
            harness.CacheMetrics(prompt_tokens=2240, cached_tokens=256),
            min_cached_tokens=2048,
            suffix_prefill_limit=256,
        )
        self.assertFalse(ok)
        self.assertIn("below required minimum", detail)

    def test_cache_probe_fails_wrong_completion_even_with_cached_tokens(self):
        harness = load_module()
        original_post_json = harness.post_json
        responses = [
            {
                "choices": [{"message": {"content": f"warmup includes {harness.KV_PIN}"}}],
                "usage": {
                    "prompt_tokens": 2240,
                    "prompt_tokens_details": {"cached_tokens": 2176},
                },
            },
            {
                "choices": [{"message": {"content": "wrong cached completion"}}],
                "usage": {
                    "prompt_tokens": 2240,
                    "prompt_tokens_details": {"cached_tokens": 2176},
                },
            },
        ]

        def fake_post_json(base_url, path, payload, timeout):
            del base_url, path, payload, timeout
            return responses.pop(0), 200

        harness.post_json = fake_post_json
        try:
            result = harness.run_cache_probe(
                base_url="http://localhost:9337/v1",
                model="direct-model",
                phase="same_prefix_cache",
                timeout=180.0,
                min_cached_tokens=2048,
                suffix_prefill_limit=256,
            )
        finally:
            harness.post_json = original_post_json

        self.assertFalse(result.ok)
        self.assertIn("response missing expected values", result.detail)
        self.assertEqual(responses, [])

    def test_log_scan_detects_memory_slot_and_eviction_errors(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "skippy-native.log"
            log_path.write_text(
                "Grammar triggered on regex: '<tool_call>'\n"
                "decode: failed to find a memory slot for batch of size 2048\n"
                "skippy.kv.decision proactive_eviction status=error\n",
                encoding="utf-8",
            )

            findings = harness.scan_failure_logs([log_path])

        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0].pattern, "failed to find a memory slot")
        self.assertEqual(findings[1].pattern, "proactive_eviction status=error")

    def test_native_log_scan_ignores_preexisting_failures_after_checkpoint(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "skippy-native.log"
            log_path.write_text(
                "old llama_decode failed before this certification\n",
                encoding="utf-8",
            )
            checkpoints = harness.capture_native_log_checkpoints([log_path])
            log_path.write_text(
                log_path.read_text(encoding="utf-8") + "new benign line\n",
                encoding="utf-8",
            )

            findings = harness.scan_failure_logs_since(checkpoints)

        self.assertEqual(findings, [])

    def test_native_log_scan_reports_failures_appended_after_checkpoint(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "skippy-native.log"
            log_path.write_text("old benign line\n", encoding="utf-8")
            checkpoints = harness.capture_native_log_checkpoints([log_path])
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("decode: failed to find a memory slot for batch of size 2048\n")

            findings = harness.scan_failure_logs_since(checkpoints)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line_number, 2)
        self.assertEqual(findings[0].pattern, "failed to find a memory slot")

    def test_native_log_scan_preserves_file_line_number_after_checkpoint(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "skippy-native.log"
            log_path.write_text(
                "old benign line 1\n"
                "old llama_decode failed before this certification\n"
                "old benign line 3\n",
                encoding="utf-8",
            )
            checkpoints = harness.capture_native_log_checkpoints([log_path])
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("new benign line 4\n")
                handle.write("RuntimeError: llama_decode failed\n")

            findings = harness.scan_failure_logs_since(checkpoints)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line_number, 5)
        self.assertEqual(findings[0].pattern, "RuntimeError: llama_decode failed")

    def test_native_log_scan_reports_new_log_created_after_checkpoint(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "skippy-native.log"
            checkpoints = harness.capture_native_log_checkpoints([log_path])
            log_path.write_text("RuntimeError: llama_decode failed\n", encoding="utf-8")

            findings = harness.scan_failure_logs_since(checkpoints)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].pattern, "RuntimeError: llama_decode failed")

    def test_native_log_scan_rescans_truncated_log_after_checkpoint(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "skippy-native.log"
            log_path.write_text("old benign line that makes the file long\n", encoding="utf-8")
            checkpoints = harness.capture_native_log_checkpoints([log_path])
            log_path.write_text(
                "skippy.kv.decision proactive_eviction status=error\n",
                encoding="utf-8",
            )

            findings = harness.scan_failure_logs_since(checkpoints)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].pattern, "proactive_eviction status=error")

    def test_prepare_transcript_dir_removes_stale_attempt_files(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript_dir = Path(tmp) / "transcripts"
            transcript_dir.mkdir()
            stale = transcript_dir / "direct-model-attempt-1.jsonl"
            stale.write_text('{"phase":"old"}\n', encoding="utf-8")

            harness.prepare_transcript_dir(transcript_dir)

            self.assertTrue(transcript_dir.is_dir())
            self.assertFalse(stale.exists())

    def test_run_certification_resets_transcripts_before_writing(self):
        harness = load_module()
        original_tool_loop = harness.run_tool_loop_probe
        original_cache_probe = harness.run_cache_probe

        def fake_tool_loop(
            base_url,
            model,
            attempt,
            timeout,
            pressure_turns,
            transcript_dir,
        ):
            del base_url, timeout, pressure_turns
            harness.record_transcript(
                transcript_dir / f"{model}-attempt-{attempt}.jsonl",
                "fresh",
                200,
            )
            return harness.ProbeResult(
                model=model,
                attempt=attempt,
                phase="tool_loop",
                ok=True,
                detail="fresh transcript",
                elapsed_ms=1,
            )

        def fake_cache_probe(
            base_url,
            model,
            phase,
            timeout,
            min_cached_tokens,
            suffix_prefill_limit,
        ):
            del base_url, timeout, min_cached_tokens, suffix_prefill_limit
            return harness.ProbeResult(
                model=model,
                attempt=0,
                phase=phase,
                ok=True,
                detail="cache ok",
                elapsed_ms=1,
            )

        harness.run_tool_loop_probe = fake_tool_loop
        harness.run_cache_probe = fake_cache_probe
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                transcript_dir = output_dir / "transcripts"
                transcript_dir.mkdir()
                stale = transcript_dir / "direct-model-attempt-1.jsonl"
                obsolete = transcript_dir / "old-model-attempt-3.jsonl"
                stale.write_text('{"phase":"old"}\n', encoding="utf-8")
                obsolete.write_text('{"phase":"obsolete"}\n', encoding="utf-8")

                harness.run_certification(
                    base_url="http://localhost:9337/v1",
                    models=["direct-model"],
                    attempts=1,
                    timeout=180.0,
                    pressure_turns=8,
                    min_cached_tokens=2048,
                    suffix_prefill_limit=256,
                    native_logs=[],
                    output_dir=output_dir,
                )

                lines = stale.read_text(encoding="utf-8").splitlines()
                payload = json.loads(lines[0])

            self.assertEqual(len(lines), 1)
            self.assertEqual(payload["phase"], "fresh")
            self.assertFalse(obsolete.exists())
        finally:
            harness.run_tool_loop_probe = original_tool_loop
            harness.run_cache_probe = original_cache_probe

    def test_record_transcript_appends_within_attempt(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            transcript_path = Path(tmp) / "transcripts" / "direct-model-attempt-1.jsonl"
            harness.record_transcript(transcript_path, "first_tool_call", 200)
            harness.record_transcript(transcript_path, "pressure_turn_1", 200)

            phases = [
                json.loads(line)["phase"]
                for line in transcript_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(phases, ["first_tool_call", "pressure_turn_1"])

    def test_write_evidence_outputs_manifest_results_and_summary(self):
        harness = load_module()
        plan = harness.build_plan(
            base_url="http://localhost:9337/v1",
            models=["direct-model"],
            attempts=1,
            pressure_turns=8,
            timeout=180.0,
            output_dir=Path("target/kv-tool-loop-stability/latest"),
            min_cached_tokens=2048,
            suffix_prefill_limit=256,
            native_logs=[],
        )
        results = [
            harness.ProbeResult(
                model="direct-model",
                attempt=1,
                phase="tool_loop",
                ok=True,
                detail="tool loop completed",
                elapsed_ms=25,
                status_code=200,
                prompt_tokens=None,
                cached_tokens=None,
            ),
            harness.ProbeResult(
                model="direct-model",
                attempt=1,
                phase="same_prefix_cache",
                ok=False,
                detail="cached_tokens=256 below required minimum 2048",
                elapsed_ms=12,
                status_code=200,
                prompt_tokens=2240,
                cached_tokens=256,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            harness.write_evidence(output_dir, plan, results)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            result_lines = (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(manifest["name"], "kv-tool-loop-stability")
        self.assertEqual(manifest["pressure_turns"], 8)
        self.assertEqual(manifest["suffix_prefill_limit"], 256)
        self.assertEqual(manifest["timeout_seconds"], 180.0)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(result_lines), 2)
        self.assertIn("KV Tool-Loop Stability Summary", summary_md)
        self.assertIn("same_prefix_cache", summary_md)


if __name__ == "__main__":
    unittest.main()
