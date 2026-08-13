from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class ReusableWorkflowRunnerTrustTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_reusable_workflows_accept_semantic_inputs_not_raw_runners(self) -> None:
        offenders = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            workflow = path.read_text(encoding="utf-8")
            if "workflow_call:" not in workflow:
                continue
            if "fromJson(inputs.runs_on)" in workflow or "runs_on:" in workflow:
                offenders.append(path.name)
        self.assertEqual([], offenders)

    def test_credential_smokes_remain_fixed_github_hosted(self) -> None:
        for name in (
            "hf-download-smoke.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "sdk-smoke.yml",
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn("ubuntu-24.04", workflow)
                self.assertNotIn("depot-ubuntu", workflow)
                checkout_count = workflow.count("uses: actions/checkout@")
                persisted_credentials_disabled = workflow.count(
                    "persist-credentials: false"
                )
                self.assertEqual(
                    checkout_count,
                    persisted_credentials_disabled,
                )

    def test_main_entry_and_orchestrator_do_not_use_pull_request_target(self) -> None:
        for name in (
            "ci.yml",
            "pr_builds.yml",
            "ci-control.yml",
            "ci-orchestrator.yml",
        ):
            with self.subTest(workflow=name):
                self.assertNotIn("pull_request_target", self.workflow(name))

    def test_main_macos_and_windows_slices_have_fixed_platforms(self) -> None:
        platform_slices = {
            "macos": (
                "macos-15",
                (
                    "ci-macos-host-slice.yml",
                    "ci-macos-runtime-slice.yml",
                    "ci-macos-product-slice.yml",
                ),
            ),
            "windows": (
                "windows-2022",
                (
                    "ci-windows-host-slice.yml",
                    "ci-windows-runtime-slice.yml",
                    "ci-windows-product-slice.yml",
                ),
            ),
        }
        for platform, (runner, names) in platform_slices.items():
            for name in names:
                with self.subTest(platform=platform, workflow=name):
                    workflow = self.workflow(name)
                    self.assertIn(f"runs-on: {runner}", workflow)
                    self.assertNotIn("macos-latest", workflow)
                    self.assertNotIn("windows-latest", workflow)

    def test_main_runner_contract_preserves_image_checks(self) -> None:
        workflow = self.workflow("ci-runner-contract-slice.yml")
        self.assertIn("inputs.profile == 'main'", workflow)
        self.assertIn("runner: mesh-llm-amd64", workflow)
        self.assertIn("runner: mesh-llm-arm64", workflow)
        self.assertIn("verify-runner-image self-hosted", workflow)

    def test_runner_contract_rejects_malformed_roles_and_missing_workflows(self) -> None:
        workflow = self.workflow("ci-runner-contract-slice.yml")
        self.assertIn('.runner_roles | type == "object"', workflow)
        self.assertIn('.value | type == "string"', workflow)
        self.assertIn('test("depot"; "i")', workflow)
        self.assertIn('[[ -f "$workflow" ]]', workflow)
        self.assertIn('workflow_scan_targets+=("$workflow")', workflow)
        self.assertIn(
            '[[ "$workflow" != ".github/workflows/ci-runner-contract-slice.yml" ]]',
            workflow,
        )
        self.assertIn('"${workflow_scan_targets[@]}"', workflow)
        self.assertIn(
            "grep -nE '^[[:space:]]+pull_request_target:'",
            workflow,
        )
        self.assertNotIn("2>/dev/null", workflow)

    def test_sdk_slice_matches_parsed_row_ids(self) -> None:
        linux = self.workflow("ci-linux-sdk-slice.yml")
        macos = self.workflow("ci-macos-sdk-slice.yml")
        self.assertEqual(
            linux.count("contains(fromJson(inputs.sdk_matrix).*.id"),
            2,
        )
        self.assertEqual(
            macos.count("contains(fromJson(inputs.sdk_matrix).*.id"),
            1,
        )
        self.assertNotIn("contains(inputs.sdk_matrix", linux + macos)

    def test_depot_allowlist_excludes_credential_smokes(self) -> None:
        migration = (ROOT / "ci" / "DEPOT_MIGRATION.md").read_text(
            encoding="utf-8",
        )
        allowlist_start = migration.index("The current main allowlist is:")
        allowlist_end = migration.index("```", allowlist_start)
        allowlist_end = migration.index("```", allowlist_end + 3)
        allowlist = migration[allowlist_start:allowlist_end]
        self.assertIn("ci-linux-lane.yml@refs/heads/main", allowlist)
        self.assertIn("ci-linux-runtime-slice.yml@refs/heads/main", allowlist)
        self.assertIn("depot-canary.yml@refs/heads/main", allowlist)
        self.assertIn("release.yml@refs/heads/main", allowlist)
        for name in (
            "hf-download-smoke.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "sdk-smoke.yml",
            "swift-sdk-artifact.yml",
        ):
            with self.subTest(workflow=name):
                self.assertNotIn(name, allowlist)

    def test_pr_entrypoint_maps_no_repository_secret(self) -> None:
        workflow = self.workflow("pr_builds.yml")
        self.assertNotIn("secrets:", workflow)
        self.assertNotIn("HF_TOKEN", workflow)

    def test_pr_facing_checkouts_disable_persisted_credentials(self) -> None:
        names = [
            "docker-precheck.yml",
            "ci-control.yml",
            "ci-orchestrator.yml",
            "static-abi-artifact.yml",
            *sorted(
                path.name
                for pattern in ("ci-*-slice.yml", "ci-*-lane.yml")
                for path in WORKFLOWS.glob(pattern)
            ),
        ]
        for name in names:
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                checkout_count = workflow.count("uses: actions/checkout@")
                if checkout_count:
                    self.assertEqual(
                        checkout_count,
                        workflow.count("persist-credentials: false"),
                    )

    def test_future_pr_depot_executor_requires_least_privilege_token_handling(
        self,
    ) -> None:
        migration = (ROOT / "ci" / "DEPOT_MIGRATION.md").read_text(
            encoding="utf-8",
        )
        section_start = migration.index("## Future protected PR Depot executor")
        section_end = migration.index("\n## ", section_start + 4)
        section = migration[section_start:section_end]
        self.assertIn("`permissions: contents: read`", section)
        self.assertIn("`persist-credentials: false`", section)


if __name__ == "__main__":
    unittest.main()
