from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PLATFORM = WORKFLOWS / "ci-platform-checks-slice.yml"


class PrWorkflowArtifactTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text()

    def test_windows_log_store_privacy_checks_are_platform_owned(self):
        workflow = PLATFORM.read_text()
        self.assertIn("name: Test Windows log artifact privacy ACL", workflow)
        self.assertIn(
            "windows_artifact_paths_have_current_owner_and_exact_user_only_dacl",
            workflow,
        )
        self.assertIn("name: Test Windows log SQLite storage ACL", workflow)
        self.assertIn(
            "sqlite_root_database_and_sidecars_have_only_current_user_acl",
            workflow,
        )

    def test_pr_entrypoint_calls_the_protected_native_composer(self):
        workflow = self.workflow("pr_builds.yml")
        self.assertIn("pull_request:", workflow)
        self.assertIn("name: PR Validation", workflow)
        self.assertIn(
            "uses: Mesh-LLM/mesh-llm/.github/workflows/ci-orchestrator.yml@main",
            workflow,
        )
        self.assertNotIn("actions.createWorkflowDispatch", workflow)
        self.assertNotIn("prepare-host-input", workflow)

    def test_controller_dispatches_only_named_topic_and_platform_workflows(self):
        workflow = self.workflow("ci-control.yml")
        for lane in ("quality", "website", "linux", "macos", "windows"):
            self.assertIn(f"'ci-{lane}-lane.yml'", workflow)
        self.assertEqual(1, workflow.count("uses: ./.github/actions/plan-ci"))
        self.assertIn("name: 'CI Required'", workflow)
        self.assertIn("ci-orchestrator.yml@main", self.workflow("pr_builds.yml"))
        self.assertNotIn("ci-orchestrator.yml", self.workflow("ci.yml"))

    def test_platform_consumers_require_only_matching_producers(self):
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                lane = self.workflow(f"ci-{platform}-lane.yml")
                self.assertIn("needs: [hosts, native_runtimes]", lane)
                self.assertIn("needs.hosts.result == 'success'", lane)
                self.assertIn("needs.native_runtimes.result == 'success'", lane)
                native_start = lane.index("  native_runtimes:")
                product_start = lane.index("  runtime_product:")
                self.assertLess(native_start, product_start)
                native = lane[native_start:product_start]
                self.assertNotIn("needs.hosts", native)

    def test_host_slices_are_platform_pure(self):
        expected = {
            "linux": ("Linux", "ci-host-linux-", "macOS host", "Windows host"),
            "macos": ("macOS", "ci-host-macos-", "Linux host", "Windows host"),
            "windows": ("Windows", "ci-host-windows-", "Linux host", "macOS host"),
        }
        for platform, (label, artifact, other_a, other_b) in expected.items():
            with self.subTest(platform=platform):
                workflow = self.workflow(f"ci-{platform}-host-slice.yml")
                self.assertIn(f"name: {label} host (", workflow)
                self.assertIn(f"name: {artifact}", workflow)
                self.assertIn("name: ${{ inputs.ui_artifact_name }}", workflow)
                self.assertNotIn(other_a, workflow)
                self.assertNotIn(other_b, workflow)
        windows = self.workflow("ci-windows-host-slice.yml")
        self.assertIn("prepare-windows-host-input", windows)
        self.assertNotIn("build-windows.ps1", windows)

    def test_runtime_producers_and_product_composers_are_separate(self):
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                runtime = self.workflow(f"ci-{platform}-runtime-slice.yml")
                product = self.workflow(f"ci-{platform}-product-slice.yml")
                self.assertIn("prepare-native-runtime-input", runtime)
                self.assertNotIn("compose-product-input", runtime)
                self.assertIn("compose-product-input", product)
                self.assertNotIn("prepare-native-runtime-input", product)
                self.assertNotIn("cargo build", product)
                self.assertNotIn("compose_products", runtime + product)

    def test_control_plane_fail_open_executes_both_web_rows(self):
        workflow = self.workflow("ci-website-lane.yml")
        control_domain = "contains(fromJson(inputs.lane_plan_json).domains, 'ci-control')"
        self.assertIn(
            "ui_changed: ${{ fromJson(inputs.lane_plan_json).signals.ui_changed || "
            f"{control_domain} }}}}",
            workflow,
        )
        self.assertIn(
            "website_changed: ${{ fromJson(inputs.lane_plan_json).signals.website_changed || "
            f"{control_domain} }}}}",
            workflow,
        )

    def test_rust_test_batches_isolate_cargo_feature_resolution(self):
        workflow = self.workflow("ci-rust-tests-slice.yml")
        self.assertIn('cargo test --locked -p "$crate"', workflow)
        self.assertNotIn('args+=("-p" "$crate")', workflow)

    def test_full_swift_sdk_has_a_cold_native_build_budget(self):
        workflow = self.workflow("ci-macos-lane.yml")
        self.assertIn("timeout_minutes: 90", workflow)


if __name__ == "__main__":
    unittest.main()
