from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class ReusableWorkflowRunnerTrustTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_no_reusable_workflow_allocates_a_raw_caller_label(self) -> None:
        offenders = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            workflow = path.read_text(encoding="utf-8")
            if "workflow_call:" not in workflow:
                continue
            if "fromJson(inputs.runs_on)" in workflow:
                offenders.append(path.name)

        self.assertEqual([], offenders)

    def test_credential_bearing_smokes_are_github_hosted(self) -> None:
        for name in (
            "hf-download-smoke.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn("runs-on: ubuntu-24.04", workflow)
                self.assertNotIn("runs_on:", workflow)
                self.assertNotIn("depot-ubuntu", workflow)
                self.assertIn("persist-credentials: false", workflow)

    def test_sdk_and_swift_runners_are_fixed_hosted_labels(self) -> None:
        sdk_smoke = self.workflow("sdk-smoke.yml")
        swift_producer = self.workflow("swift-sdk-artifact.yml")

        for workflow in (sdk_smoke, swift_producer):
            self.assertIn("macos-15", workflow)
            self.assertNotIn("macos_runner:", workflow)
            self.assertNotIn("macos-latest", workflow)
            self.assertNotIn("runs_on:", workflow)
            self.assertNotIn("fromJson(inputs.", workflow)
            self.assertNotIn("depot-ubuntu", workflow)
            self.assertNotIn("self-hosted", workflow)

        self.assertIn("'ubuntu-24.04'", sdk_smoke)
        self.assertIn("'ubuntu-24.04-arm'", sdk_smoke)
        self.assertIn(
            "inputs.kotlin_artifact_target == "
            "'aarch64-unknown-linux-gnu'",
            sdk_smoke,
        )
        self.assertIn(
            "x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu",
            sdk_smoke,
        )

    def test_main_macos_product_graph_uses_the_swift_toolchain_boundary(
        self,
    ) -> None:
        main = self.workflow("ci.yml")

        self.assertNotIn("macos-latest", main)
        for job_id in (
            "macos_host_input",
            "macos_metal_runtime_input",
            "macos_cpu_artifact",
            "macos_unit_tests",
        ):
            with self.subTest(job=job_id):
                match = re.search(
                    rf"(?ms)^  {re.escape(job_id)}:\n"
                    r"(?P<body>.*?)(?=^  [A-Za-z0-9_]+:\n|\Z)",
                    main,
                )
                self.assertIsNotNone(match)
                self.assertIn(
                    "runs-on: macos-15",
                    match.group("body") if match else "",
                )

    def test_nightly_reusable_workflow_cannot_select_a_runner(self) -> None:
        wrapper = self.workflow("nightly-stability.yml")
        reusable = self.workflow("nightly-stability-run.yml")

        self.assertNotIn("runs_on:", wrapper)
        self.assertNotIn("MESH_NIGHTLY_STABILITY_RUNS_ON", wrapper)
        self.assertNotIn("runs_on:", reusable)
        self.assertIn("runs-on: ubuntu-24.04", reusable)
        self.assertIn("persist-credentials: false", reusable)

    def test_depot_allowlist_excludes_credential_smokes(self) -> None:
        migration = (ROOT / "ci" / "DEPOT_MIGRATION.md").read_text(
            encoding="utf-8",
        )
        allowlist_start = migration.index("The initial main allowlist is:")
        allowlist_end = migration.index("```", allowlist_start)
        allowlist_end = migration.index("```", allowlist_end + 3)
        allowlist = migration[allowlist_start:allowlist_end]

        self.assertIn("native-sdk-artifact.yml@refs/heads/main", allowlist)
        self.assertIn("static-abi-artifact.yml@refs/heads/main", allowlist)
        for name in (
            "hf-download-smoke.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "sdk-smoke.yml",
            "swift-sdk-artifact.yml",
        ):
            with self.subTest(workflow=name):
                self.assertNotIn(name, allowlist)

    def test_pull_request_builds_do_not_receive_hugging_face_secret(self) -> None:
        pr_builds = self.workflow("pr_builds.yml")

        self.assertNotIn("secrets.HF_TOKEN", pr_builds)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN:", pr_builds)

    def test_pr_facing_checkouts_do_not_persist_job_credentials(self) -> None:
        for name in (
            "docker-precheck.yml",
            "pr_quality.yml",
            "pr_website.yml",
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                checkout_count = workflow.count("uses: actions/checkout@")
                self.assertGreater(checkout_count, 0)
                self.assertEqual(
                    checkout_count,
                    workflow.count("persist-credentials: false"),
                )


if __name__ == "__main__":
    unittest.main()
