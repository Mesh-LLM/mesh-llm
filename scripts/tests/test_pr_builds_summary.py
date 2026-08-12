from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class RequiredSummaryTests(unittest.TestCase):
    def test_orchestrator_summary_is_stable_and_cancellation_safe(self):
        workflow = (ROOT / ".github/workflows/ci-orchestrator.yml").read_text()
        self.assertIn("name: CI Required", workflow)
        self.assertIn("if: ${{ !cancelled() }}", workflow)
        self.assertNotIn("always()", workflow)

    def test_summary_directly_needs_static_superset(self):
        workflow = (ROOT / ".github/workflows/ci-orchestrator.yml").read_text()
        summary_start = workflow.index("  summary:")
        next_job = re.search(
            r"(?m)^  [A-Za-z0-9_]+:\s*$",
            workflow[summary_start + len("  summary:") :],
        )
        summary_end = (
            summary_start + len("  summary:") + next_job.start()
            if next_job
            else len(workflow)
        )
        summary = workflow[
            summary_start:summary_end
        ]
        for job in (
            "plan",
            "quality",
            "web",
            "ui_artifact",
            "static_abi",
            "rust_tests",
            "hosts_linux",
            "hosts_macos",
            "hosts_windows",
            "native_runtimes_linux",
            "native_runtimes_macos",
            "native_runtimes_windows",
            "runtime_product_linux",
            "runtime_product_macos",
            "runtime_product_windows",
            "platform_checks",
            "product_smoke",
            "sdk",
            "runner_contract",
        ):
            self.assertIn(f"      - {job}", summary)

    def test_summary_rejects_bad_results(self):
        workflow = (ROOT / ".github/workflows/ci-orchestrator.yml").read_text()
        self.assertIn(
            'if .key == "plan"\n              then .value.result == "success"',
            workflow,
        )
        self.assertIn('result == "success"', workflow)
        self.assertIn('result == "skipped"', workflow)
        self.assertIn("set -euo pipefail", workflow)
        self.assertIn("jq -e --argjson required", workflow)


if __name__ == "__main__":
    unittest.main()
