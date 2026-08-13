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

    def test_pr_entrypoints_call_protected_native_lanes(self):
        for lane in ("quality", "website", "linux", "macos", "windows"):
            workflow = self.workflow(f"pr_{lane}.yml")
            self.assertIn("pull_request:", workflow)
            self.assertIn(
                f"uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main",
                workflow,
            )
            self.assertNotIn("actions.createWorkflowDispatch", workflow)
            self.assertNotIn("prepare-host-input", workflow)

    def test_legacy_pr_entrypoint_is_an_inert_migration_shim(self):
        workflow = self.workflow("pr_builds.yml")
        self.assertIn("workflow_call:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("ci-orchestrator.yml", workflow)

    def test_pr_validation_has_exactly_five_focused_entrypoints(self):
        expected = {
            "pr_quality.yml": "quality",
            "pr_website.yml": "website",
            "pr_linux.yml": "linux",
            "pr_macos.yml": "macos",
            "pr_windows.yml": "windows",
        }
        actual = {
            path.name
            for path in WORKFLOWS.glob("*.yml")
            if "\n  pull_request:\n" in path.read_text()
        }
        self.assertEqual(set(expected), actual)
        self.assertFalse((WORKFLOWS / "ci-orchestrator.yml").exists())

        for filename, lane in expected.items():
            workflow = self.workflow(filename)
            protected_calls = [
                line.strip()
                for line in workflow.splitlines()
                if line.strip().startswith("uses: Mesh-LLM/mesh-llm/.github/workflows/ci-")
            ]
            self.assertEqual(
                [f"uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main"],
                protected_calls,
            )
            self.assertNotIn("paths:", workflow)
            self.assertNotIn("createWorkflowDispatch", workflow)

    def test_ci_docs_forbid_monolithic_or_dispatch_only_pr_visibility(self):
        docs = (ROOT / "ci" / "ci.md").read_text()
        self.assertIn("The five-way split is a hard CI architecture invariant", docs)
        self.assertIn("`dispatched`, with the real work detached", docs)
        self.assertIn("Do not reintroduce", docs)

    def test_controller_dispatches_only_named_topic_and_platform_workflows(self):
        workflow = self.workflow("ci-control.yml")
        for lane in ("quality", "website", "linux", "macos", "windows"):
            self.assertIn(f"'ci-{lane}-lane.yml'", workflow)
        self.assertEqual(1, workflow.count("uses: ./.github/actions/plan-ci"))
        self.assertIn("name: 'CI Required'", workflow)
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

    def test_pr_platform_critical_matrices_fail_fast_by_profile(self):
        slices = (
            "ci-rust-tests-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
            "ci-macos-host-slice.yml",
            "ci-macos-runtime-slice.yml",
            "ci-macos-product-slice.yml",
            "ci-windows-host-slice.yml",
            "ci-windows-runtime-slice.yml",
            "ci-windows-product-slice.yml",
            "ci-platform-checks-slice.yml",
        )
        for filename in slices:
            with self.subTest(filename=filename):
                workflow = self.workflow(filename)
                self.assertIn("fail_fast:", workflow)
                self.assertIn("fail-fast: ${{ inputs.fail_fast }}", workflow)

        for platform in ("linux", "macos", "windows"):
            lane = self.workflow(f"ci-{platform}-lane.yml")
            self.assertIn(
                "fail_fast: ${{ inputs.original_event_name == 'pull_request' }}",
                lane,
            )

        quality = self.workflow("ci-quality-slice.yml")
        self.assertIn("fail-fast: false", quality)
        self.assertNotIn("fail-fast: ${{ inputs.fail_fast }}", quality)

    def test_pr_cache_publishers_are_exact_and_bounded(self):
        ui_artifact = self.workflow("ci-ui-artifact-slice.yml")
        website = self.workflow("ci-web-slice.yml")
        self.assertNotIn("name: Save pnpm store", ui_artifact)
        self.assertEqual(1, website.count("name: Save pnpm store"))
        self.assertIn("cache: npm", website)
        self.assertIn("website/package-lock.json", website)

        windows = self.workflow("ci-windows-runtime-slice.yml")
        self.assertIn("name: Save exact PR-scoped Windows ABI build", windows)
        self.assertIn("key: ${{ steps.llama_cache.outputs.cache-primary-key }}", windows)
        self.assertNotIn("restore-keys:", windows)

        platform = self.workflow("ci-platform-checks-slice.yml")
        self.assertIn("inputs.original_event_name == 'pull_request'", platform)
        self.assertIn("key: ${{ steps.llama_cache.outputs.cache-primary-key }}", platform)

        rust_tests = self.workflow("ci-rust-tests-slice.yml")
        self.assertIn("github.ref == 'refs/heads/main'", rust_tests)
        self.assertIn("original_event_name != 'pull_request'", rust_tests)


if __name__ == "__main__":
    unittest.main()
