from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class CiLaneWorkflowTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_controller_plans_once_and_dispatches_native_lane_inputs(self) -> None:
        workflow = self.workflow("ci-control.yml")
        self.assertEqual(1, workflow.count("uses: ./.github/actions/plan-ci"))
        self.assertIn("workflow_run:", workflow)
        self.assertIn("actions: write", workflow)
        self.assertIn("checks: write", workflow)
        self.assertIn("github.rest.actions.createWorkflowDispatch", workflow)
        self.assertIn("PLAN_DIGEST", workflow)
        self.assertIn("planner output digest mismatch", workflow)
        self.assertIn(
            "ref: ${{ github.event.repository.default_branch }}",
            workflow,
        )
        self.assertIn("ref: process.env.DEFAULT_BRANCH", workflow)
        self.assertNotIn("ref: process.env.SOURCE_REF", workflow)
        self.assertNotIn("ref: process.env.SOURCE_SHA", workflow)
        self.assertNotIn("download-artifact", workflow)
        for name in ("quality", "website", "linux", "macos", "windows"):
            self.assertIn(f"ci-{name}-lane.yml", workflow)

    def test_controller_does_not_dispatch_bootstrap_or_fork_runs(self) -> None:
        workflow = self.workflow("ci-control.yml")
        self.assertIn("job.name === 'Bootstrap PR CI'", workflow)
        self.assertIn("job.name.startsWith('Bootstrap PR CI / ')", workflow)
        self.assertIn("job.name.startsWith('Bootstrap main CI / ')", workflow)
        self.assertIn("job.conclusion !== 'skipped'", workflow)
        self.assertIn("head_repository.full_name == github.repository", workflow)
        self.assertIn("should_dispatch", workflow)

        pr_workflow = self.workflow("pr_builds.yml")
        self.assertIn("github.rest.pulls.listFiles", pr_workflow)
        self.assertIn("controlPlaneChanged", pr_workflow)
        self.assertIn("filename.startsWith('.github/')", pr_workflow)
        self.assertIn(
            "context.eventName === 'workflow_dispatch' || controlPlaneChanged",
            pr_workflow,
        )
        self.assertIn("pull.head.repo?.full_name", pr_workflow)
        self.assertIn(
            "ref: context.payload.repository.default_branch",
            pr_workflow,
        )

    def test_thin_routes_do_not_compete_with_bootstrap_concurrency(self) -> None:
        for name in ("pr_builds.yml", "ci.yml"):
            with self.subTest(workflow=name):
                self.assertNotIn("concurrency:", self.workflow(name))
        self.assertIn("concurrency:", self.workflow("ci-orchestrator.yml"))

    def test_lane_workflows_support_bootstrap_calls_and_native_dispatch(self) -> None:
        checks = {
            "quality": "CI / Quality",
            "website": "CI / Website",
            "linux": "CI / Linux",
            "macos": "CI / macOS",
            "windows": "CI / Windows",
        }
        for lane, check in checks.items():
            with self.subTest(lane=lane):
                workflow = self.workflow(f"ci-{lane}-lane.yml")
                self.assertIn("workflow_call:", workflow)
                self.assertIn("workflow_dispatch:", workflow)
                self.assertIn("lane_plan_json:", workflow)
                self.assertIn(f"name: {check}", workflow)
                self.assertIn("uses: ./.github/actions/report-ci-lane", workflow)
                self.assertIn(
                    "ref: ${{ github.event.repository.default_branch }}",
                    workflow,
                )

    def test_dispatched_lanes_pass_source_sha_only_to_product_workflows(
        self,
    ) -> None:
        lane_workflows = {
            "ci-quality-lane.yml": 3,
            "ci-website-lane.yml": 2,
            "ci-linux-lane.yml": 10,
            "ci-macos-lane.yml": 9,
            "ci-windows-lane.yml": 6,
        }
        for workflow_name, expected_calls in lane_workflows.items():
            with self.subTest(workflow=workflow_name):
                workflow = self.workflow(workflow_name)
                self.assertEqual(
                    expected_calls,
                    workflow.count("source_sha: ${{ inputs.source_sha }}"),
                )

        product_workflows = (
            "ci-quality-slice.yml",
            "ci-runner-contract-slice.yml",
            "ci-web-slice.yml",
            "ci-ui-artifact-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-host-slice.yml",
            "ci-runtime-product-slice.yml",
            "ci-platform-checks-slice.yml",
            "static-abi-artifact.yml",
            "native-sdk-artifact.yml",
            "swift-sdk-artifact.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "sdk-smoke.yml",
            "hf-download-smoke.yml",
        )
        for workflow_name in product_workflows:
            with self.subTest(workflow=workflow_name):
                workflow = self.workflow(workflow_name)
                self.assertIn("source_sha:", workflow)
                self.assertIn(
                    "ref: ${{ inputs.source_sha || github.sha }}",
                    workflow,
                )

    def test_lane_plans_are_bounded_platform_projections(self) -> None:
        action = (ROOT / ".github/actions/plan-ci/action.yml").read_text(
            encoding="utf-8"
        )
        for output in (
            "quality_lane_plan",
            "website_lane_plan",
            "linux_lane_plan",
            "macos_lane_plan",
            "windows_lane_plan",
        ):
            self.assertIn(f"{output}:", action)
            self.assertIn(f'echo "{output}=$', action)
        for platform in ("linux", "macos", "windows"):
            self.assertIn(f'select(.platform == "{platform}")', action)

    def test_topic_lane_projections_are_valid_jq(self) -> None:
        action = (ROOT / ".github/actions/plan-ci/action.yml").read_text(
            encoding="utf-8"
        )
        plans = (
            {
                "profile": "pr-ready",
                "domains": ["ci-control"],
                "required_slices": ["quality", "runner-contract", "web"],
                "signals": {"rust_changed": True},
                "budgets": {"total_max_workers": 10},
                "matrices": {"clippy": [{"id": "batch-0"}]},
            },
            {
                "profile": "pr-ready",
                "domains": ["docs"],
                "required_slices": [],
                "signals": {"rust_changed": False},
                "budgets": {"total_max_workers": 2},
                "matrices": {"clippy": []},
            },
        )
        for output, lane in (
            ("quality_lane_plan", "quality"),
            ("website_lane_plan", "website"),
        ):
            match = re.search(
                rf"{output}=\$\(jq -ce '(.*?)' ci-plan\.json\)",
                action,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            for required, plan in zip((True, False), plans, strict=True):
                with self.subTest(output=output, required=required):
                    result = subprocess.run(
                        ["jq", "-ce", match.group(1)],
                        input=json.dumps(plan),
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    projection = json.loads(result.stdout)
                    self.assertEqual(lane, projection["lane"])
                    self.assertIs(required, projection["required"])
                    self.assertEqual(
                        plan["required_slices"], projection["required_slices"]
                    )
                    self.assertEqual(plan["signals"], projection["signals"])
                    self.assertEqual(plan["budgets"], projection["budgets"])
                    expected_matrices = (
                        {"clippy": plan["matrices"]["clippy"]}
                        if lane == "quality"
                        else {}
                    )
                    self.assertEqual(expected_matrices, projection["matrices"])

    def test_dispatched_pr_lanes_receive_no_hugging_face_credential(self) -> None:
        controller = self.workflow("ci-control.yml")
        self.assertNotIn("HF_TOKEN", controller)
        for name in ("ci-linux-lane.yml", "ci-macos-lane.yml"):
            workflow = self.workflow(name)
            self.assertIn("inputs.original_event_name == 'push'", workflow)

    def test_dispatched_main_preserves_trusted_runner_policy(self) -> None:
        selector = (ROOT / ".github/actions/select-ci-runners/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("DISPATCH_ORIGINAL_EVENT_NAME", selector)
        self.assertIn("pull_request|pull_request_target) effective_event_name=pull_request", selector)
        for name in ("static-abi-artifact.yml", "native-sdk-artifact.yml"):
            workflow = self.workflow(name)
            self.assertIn(
                "github.event.inputs.original_event_name || github.event_name",
                workflow,
            )

    def test_reporter_completes_only_correlated_checks(self) -> None:
        action = (ROOT / ".github/actions/report-ci-lane/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("check.external_id === process.env.CORRELATION_ID", action)
        self.assertIn("check.head_sha !== process.env.SOURCE_SHA", action)
        self.assertIn("if (process.env.OVERALL_CHECK_ID)", action)
        self.assertIn("github.paginate(github.rest.checks.listForRef", action)
        self.assertIn("lanes.length === expected.length", action)
        self.assertNotIn("correlated lane checks did not converge", action)

    def test_reporter_allows_protected_workflow_sha_to_differ(self) -> None:
        action = (ROOT / ".github/actions/report-ci-lane/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("context.sha", action)
        self.assertIn(
            "!/^[0-9a-f]{40}$/.test(process.env.SOURCE_SHA)",
            action,
        )
        self.assertIn("check.head_sha !== process.env.SOURCE_SHA", action)

    def test_manual_main_depot_input_is_explicitly_forwarded(self) -> None:
        main = self.workflow("ci.yml")
        orchestrator = self.workflow("ci-orchestrator.yml")
        pr = self.workflow("pr_builds.yml")

        self.assertIn(
            "use_depot: ${{ github.event_name == 'workflow_dispatch' && inputs.use_depot == true }}",
            main,
        )
        self.assertIn("use_depot:", orchestrator)
        self.assertGreaterEqual(
            orchestrator.count("use_depot: ${{ inputs.use_depot }}"),
            11,
        )
        self.assertNotIn("use_depot:", pr)
        for name in (
            "ci-quality-slice.yml",
            "ci-host-slice.yml",
            "ci-runtime-product-slice.yml",
            "static-abi-artifact.yml",
            "native-sdk-artifact.yml",
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn("use_depot:", workflow)
                self.assertIn("${{ inputs.use_depot }}", workflow)

    def test_superseded_pr_runs_cancel_by_pull_request_identity(self) -> None:
        controller = self.workflow("ci-control.yml")
        self.assertIn("core.setOutput('supersession_key', `pr-${pull.number}`)", controller)
        for lane in ("quality", "website", "linux", "macos", "windows"):
            workflow = self.workflow(f"ci-{lane}-lane.yml")
            self.assertIn("inputs.supersession_key || inputs.source_sha", workflow)


if __name__ == "__main__":
    unittest.main()
