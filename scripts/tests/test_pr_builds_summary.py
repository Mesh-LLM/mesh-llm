from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class RequiredSummaryTests(unittest.TestCase):
    def test_each_pr_entry_owns_a_native_required_job(self):
        labels = {"quality": "Quality", "website": "Website", "linux": "Linux", "macos": "macOS", "windows": "Windows"}
        for lane, label in labels.items():
            workflow = (WORKFLOWS / f"pr_{lane}.yml").read_text()
            self.assertIn(f"name: PR / {label}", workflow)
            self.assertIn("needs: [plan, lane]", workflow)
            self.assertNotIn("checks.create", workflow)

    def test_manual_plan_owns_stable_correlated_checks(self):
        workflow = (WORKFLOWS / "ci-control.yml").read_text()
        self.assertIn("name: CI · Manual Full", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("workflow_run:", workflow)
        self.assertIn("name: 'CI Required'", workflow)
        for lane in ("Quality", "Website", "Linux", "macOS", "Windows"):
            self.assertIn(f"'CI / {lane}'", workflow)
        self.assertIn("external_id: process.env.CORRELATION_ID", workflow)

    def test_each_main_entry_owns_a_native_required_job(self):
        labels = {"quality": "Quality", "website": "Website", "linux": "Linux", "macos": "macOS", "windows": "Windows"}
        for lane, label in labels.items():
            workflow = (WORKFLOWS / f"main_{lane}.yml").read_text()
            self.assertIn(f"name: Main / {label}", workflow)
            self.assertIn("needs: [plan, lane]", workflow)
            self.assertNotIn("checks.create", workflow)

    def test_each_lane_has_one_cancellation_safe_summary(self):
        checks = {
            "quality": ("CI / Quality", ("quality", "runner_contract")),
            "website": ("CI / Website", ("web",)),
            "linux": (
                "CI / Linux",
                (
                    "ui_artifact",
                    "static_abi",
                    "rust_tests",
                    "hosts",
                    "native_runtimes",
                    "runtime_product",
                    "kotlin_sdk_input",
                    "sdk",
                    "product_smoke",
                ),
            ),
            "macos": (
                "CI / macOS",
                (
                    "validate_plan",
                    "ui_artifact",
                    "hosts",
                    "native_runtimes",
                    "runtime_product",
                    "platform_checks",
                    "swift_sdk_input",
                    "sdk",
                    "product_smoke",
                ),
            ),
            "windows": (
                "CI / Windows",
                (
                    "ui_artifact",
                    "hosts",
                    "native_runtimes",
                    "runtime_product",
                    "platform_checks",
                ),
            ),
        }
        for lane, (check_name, jobs) in checks.items():
            with self.subTest(lane=lane):
                workflow = (WORKFLOWS / f"ci-{lane}-lane.yml").read_text()
                summary_start = workflow.index("  summary:")
                summary = workflow[summary_start:]
                self.assertEqual(1, workflow.count("\n  summary:\n"))
                self.assertIn(f"    name: {check_name}", summary)
                self.assertIn("if: ${{ !cancelled() }}", summary)
                self.assertNotIn("always()", summary)
                for job in jobs:
                    self.assertRegex(
                        summary,
                        rf"(?:needs: \[[^\]]*\b{re.escape(job)}\b|      - {re.escape(job)}\n)",
                    )

    def test_lane_validator_rejects_required_skips_and_extra_work(self):
        validator = (ROOT / "scripts/validate-ci-lane-results.py").read_text()
        self.assertIn('if result != "success"', validator)
        self.assertIn('if result != "skipped"', validator)
        self.assertIn("required lane has no planned jobs", validator)


if __name__ == "__main__":
    unittest.main()
