from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / ".github/workflows/ci-orchestrator.yml"
HOST = ROOT / ".github/workflows/ci-host-slice.yml"
RUNTIME = ROOT / ".github/workflows/ci-runtime-product-slice.yml"
PLATFORM = ROOT / ".github/workflows/ci-platform-checks-slice.yml"


class PrWorkflowArtifactTests(unittest.TestCase):
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

    def test_pr_entrypoint_is_thin_and_calls_shared_orchestrator(self):
        workflow = (ROOT / ".github/workflows/pr_builds.yml").read_text()
        self.assertIn("pull_request:", workflow)
        self.assertIn("pull-requests: read", workflow)
        self.assertIn("uses: ./.github/workflows/ci-orchestrator.yml", workflow)
        self.assertNotIn("prepare-host-input", workflow)

    def test_orchestrator_owns_one_static_slice_superset_and_summary(self):
        workflow = ORCHESTRATOR.read_text()
        for slice_id in (
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
            "kotlin_sdk_input",
            "swift_sdk_input",
            "sdk_linux",
            "sdk_macos",
            "runner_contract",
        ):
            self.assertIn(f"  {slice_id}:", workflow)
        self.assertIn("name: CI Required", workflow)
        self.assertIn("NEEDS_RESULTS: ${{ toJson(needs) }}", workflow)
        self.assertIn("REQUIRED_SLICES: ${{ needs.plan.outputs.required_slices }}", workflow)
        self.assertIn("if: ${{ !cancelled() }}", workflow)

    def test_orchestrator_consumers_require_successful_producers(self):
        workflow = ORCHESTRATOR.read_text()
        for result_check in (
            "needs.static_abi.result == 'success'",
            "needs.ui_artifact.result == 'success'",
            "needs.hosts_linux.result == 'success'",
            "needs.native_runtimes_linux.result == 'success'",
            "needs.runtime_product_linux.result == 'success'",
        ):
            self.assertIn(result_check, workflow)
        self.assertIn(
            "!contains(fromJson(needs.plan.outputs.sdk_matrix).*.id, 'kotlin')",
            workflow,
        )

    def test_hosts_and_native_runtimes_are_independent_release_producers(self):
        workflow = ORCHESTRATOR.read_text()
        runtime_start = workflow.index("  native_runtimes_linux:")
        product_start = workflow.index("  runtime_product_linux:")
        runtime = workflow[runtime_start:product_start]
        product = workflow[product_start:workflow.index("  platform_checks:")]

        self.assertIn("needs: plan", runtime)
        self.assertNotIn("needs.hosts_linux", runtime)
        self.assertIn(
            "needs: [plan, hosts_linux, native_runtimes_linux]", product
        )
        self.assertIn(
            "needs: [plan, hosts_macos, native_runtimes_macos]", product
        )
        self.assertIn(
            "needs: [plan, hosts_windows, native_runtimes_windows]", product
        )
        self.assertIn("compose_products: true", product)
        self.assertIn("profile: release", workflow)
        self.assertIn("binary_target: target/release/mesh-llm", workflow)

    def test_control_plane_fail_open_executes_both_web_rows(self):
        workflow = ORCHESTRATOR.read_text()
        control_domain = "contains(needs.plan.outputs.domains, '\"ci-control\"')"
        self.assertIn(
            "ui_changed: ${{ needs.plan.outputs.ui_changed == 'true' || "
            f"{control_domain} }}}}",
            workflow,
        )
        self.assertIn(
            "website_changed: ${{ needs.plan.outputs.website_changed == 'true' || "
            f"{control_domain} }}}}",
            workflow,
        )

    def test_hosts_consume_one_ui_artifact_and_emit_platform_hosts(self):
        workflow = HOST.read_text()
        self.assertIn("name: ${{ inputs.ui_artifact_name }}", workflow)
        for platform in ("Linux", "macOS", "Windows"):
            self.assertIn(f"name: {platform} host (", workflow)
        for artifact in ("ci-host-linux-", "ci-host-macos-", "ci-host-windows-"):
            self.assertIn(f"name: {artifact}", workflow)
        self.assertIn("prepare-windows-host-input", workflow)
        self.assertNotIn("build-windows.ps1", workflow)

    def test_runtime_products_are_composition_only_consumers(self):
        workflow = RUNTIME.read_text()
        self.assertIn("compose_products:", workflow)
        self.assertIn("!inputs.compose_products", workflow)
        self.assertIn("inputs.compose_products", workflow)
        self.assertIn("prepare-native-runtime-input", workflow)
        self.assertIn("compose-product-input", workflow)
        self.assertIn("name: ci-product-linux-${{ matrix.runtime.backend }}", workflow)
        self.assertIn("name: ci-product-windows-${{ matrix.runtime.backend }}", workflow)
        product_start = workflow.index("  linux_product:")
        product_end = workflow.index("\n  macos_runtime:", product_start)
        product = workflow[product_start:product_end]
        self.assertNotIn("cargo build", product)
        self.assertNotIn("package-native-runtime.sh", product)

    def test_rust_test_batches_isolate_cargo_feature_resolution(self):
        workflow = (
            ROOT / ".github/workflows/ci-rust-tests-slice.yml"
        ).read_text()
        self.assertIn('cargo test --locked -p "$crate"', workflow)
        self.assertNotIn('args+=("-p" "$crate")', workflow)

    def test_full_swift_sdk_has_a_cold_native_build_budget(self):
        workflow = ORCHESTRATOR.read_text()
        self.assertIn("timeout_minutes: 90", workflow)


if __name__ == "__main__":
    unittest.main()
