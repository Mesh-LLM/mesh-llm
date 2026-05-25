import importlib.util
import json
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
            output_dir=Path("target/kv-tool-loop-stability/latest"),
            min_cached_tokens=2048,
            native_logs=[Path("skippy-native.log")],
        )

        self.assertEqual(plan["name"], "kv-tool-loop-stability")
        self.assertEqual(plan["base_url"], "http://localhost:9337/v1")
        self.assertEqual(plan["attempts"], 3)
        self.assertEqual(plan["min_cached_tokens"], 2048)
        self.assertIn("manifest.json", plan["evidence_files"])
        self.assertIn("results.jsonl", plan["evidence_files"])
        self.assertIn("summary.md", plan["evidence_files"])
        self.assertEqual(
            [check["phase"] for check in plan["checks"]],
            ["tool_loop", "same_prefix_cache", "exact_prefix_cache", "native_log_scan"],
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

    def test_write_evidence_outputs_manifest_results_and_summary(self):
        harness = load_module()
        plan = harness.build_plan(
            base_url="http://localhost:9337/v1",
            models=["direct-model"],
            attempts=1,
            output_dir=Path("target/kv-tool-loop-stability/latest"),
            min_cached_tokens=2048,
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
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(result_lines), 2)
        self.assertIn("KV Tool-Loop Stability Summary", summary_md)
        self.assertIn("same_prefix_cache", summary_md)


if __name__ == "__main__":
    unittest.main()
