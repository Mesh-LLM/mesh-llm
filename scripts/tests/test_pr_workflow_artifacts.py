from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PR_WORKFLOW = ROOT / ".github" / "workflows" / "pr_builds.yml"


def job_section(
    workflow: str,
    job_name: str,
    next_job_name: str | None = None,
) -> str:
    start = workflow.index(f"  {job_name}:")
    if next_job_name is None:
        return workflow[start:]
    end = workflow.index(f"  {next_job_name}:", start)
    return workflow[start:end]


class PrWorkflowArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = PR_WORKFLOW.read_text(encoding="utf-8")
        cls.host = job_section(
            cls.workflow,
            "linux_host_input",
            "linux_cpu_runtime_input",
        )
        cls.cpu_runtime = job_section(
            cls.workflow,
            "linux_cpu_runtime_input",
            "linux_cpu_artifact",
        )
        cls.cpu_product = job_section(
            cls.workflow,
            "linux_cpu_artifact",
            "linux_targets",
        )
        cls.backend_products = job_section(
            cls.workflow,
            "linux_targets",
            "rust_crate_tests",
        )
        cls.macos_host = job_section(
            cls.workflow,
            "macos_host_input",
            "macos_metal_runtime_input",
        )
        cls.macos_runtime = job_section(
            cls.workflow,
            "macos_metal_runtime_input",
            "macos_cpu_artifact",
        )
        cls.macos_product = job_section(
            cls.workflow,
            "macos_cpu_artifact",
            "swift_sdk_smoke",
        )
        cls.windows_checks = job_section(
            cls.workflow,
            "windows_checks",
            "windows_host_input",
        )
        cls.windows_host = job_section(
            cls.workflow,
            "windows_host_input",
            "windows_cpu_runtime_input",
        )
        cls.windows_cpu_runtime = job_section(
            cls.workflow,
            "windows_cpu_runtime_input",
            "windows_gpu_runtime_inputs",
        )
        cls.windows_gpu_runtimes = job_section(
            cls.workflow,
            "windows_gpu_runtime_inputs",
            "windows_cpu_product",
        )
        cls.windows_cpu_product = job_section(
            cls.workflow,
            "windows_cpu_product",
            "windows_gpu_products",
        )
        cls.windows_gpu_products = job_section(
            cls.workflow,
            "windows_gpu_products",
        )

    def test_host_profile_covers_every_backend_product_route(self) -> None:
        self.assertIn(
            "needs.changes.outputs.linux_inference_artifact_required == 'true' "
            "|| needs.changes.outputs.benchmarks == 'true'",
            self.host,
        )
        self.assertIn(
            "needs.changes.outputs.backend_changed == 'true' "
            "|| needs.changes.outputs.benchmarks == 'true'",
            self.host,
        )
        self.assertIn("&& 'release' || 'debug'", self.host)

        self.assertIn(
            "github.event_name == 'workflow_dispatch' "
            "|| needs.changes.outputs.backend_changed == 'true' "
            "|| needs.changes.outputs.benchmarks == 'true'",
            self.backend_products,
        )

    def test_cpu_runtime_only_runs_for_cpu_product_consumers(self) -> None:
        condition = (
            "if: ${{ needs.changes.outputs.linux_inference_artifact_required "
            "== 'true' && needs.changes.outputs.docs_only != 'true' }}"
        )
        self.assertIn(condition, self.cpu_runtime)
        self.assertNotIn("benchmarks", self.cpu_runtime)

    def test_cpu_product_uses_matching_immutable_inputs(self) -> None:
        self.assertIn("name: pr-linux-host-input", self.host)
        self.assertIn("name: pr-linux-cpu-runtime-input", self.cpu_runtime)
        self.assertIn(
            "needs: [changes, linux_host_input, linux_cpu_runtime_input]",
            self.cpu_product,
        )
        self.assertIn("name: pr-linux-host-input", self.cpu_product)
        self.assertIn("path: host-input", self.cpu_product)
        self.assertIn("name: pr-linux-cpu-runtime-input", self.cpu_product)
        self.assertIn("path: runtime-input", self.cpu_product)
        self.assertIn("output_dir: ci-product", self.cpu_product)
        self.assertIn(
            "path: ${{ steps.compose.outputs.archive_path }}",
            self.cpu_product,
        )

    def test_backend_products_reuse_the_same_host_artifact(self) -> None:
        self.assertIn("needs: [changes, linux_host_input]", self.backend_products)
        self.assertIn("name: pr-linux-host-input", self.backend_products)
        self.assertIn("path: host-input", self.backend_products)
        self.assertNotIn("pr-linux-release-host-input", self.workflow)
        self.assertIn("output_dir: product-input", self.backend_products)
        self.assertIn(
            "path: ${{ steps.compose.outputs.archive_path }}",
            self.backend_products,
        )

    def test_cuda_runtime_uses_the_production_multiarch_image(self) -> None:
        self.assertIn(
            "sha256:c5b85ef527230f77cf9933ef40bcb44316f9bbcb8fd2ce0651b58acda5143dfd",
            self.workflow,
        )
        self.assertNotIn(
            "sha256:295341c6c9f17c9eb69281fd454bda953799406d6915f472c914fb5f024a88ed",
            self.workflow,
        )

    def test_public_mesh_admission_is_manual_not_a_pr_gate(self) -> None:
        admission = job_section(
            self.workflow,
            "linux_public_mesh_admission",
            "hf_download_smoke",
        )

        self.assertIn("github.event_name == 'workflow_dispatch'", admission)
        self.assertNotIn("linux_client_auto_boot:", self.workflow)
        self.assertIn("scripts/ci-client-auto-test.sh", admission)
        self.assertIn("uses: ./.github/actions/restore-smoke-inputs", admission)
        self.assertIn(
            "artifact_name: ci-linux-inference-binaries",
            admission,
        )
        self.assertIn(
            "staged_binary_path: target/debug/mesh-llm",
            admission,
        )
        self.assertNotIn("uses: actions/download-artifact@", admission)
        self.assertNotIn("chmod +x target/debug/mesh-llm", admission)

    def test_linux_test_groups_use_the_same_dynamic_plan_as_main(self) -> None:
        groups = job_section(
            self.workflow,
            "linux_test_groups",
            "linux_public_mesh_admission",
        )

        self.assertIn(
            "linux_test_groups_json: "
            "${{ steps.compute.outputs.linux_test_groups_json }}",
            self.workflow,
        )
        self.assertIn(
            "needs: [changes, linux_static_abi_input]",
            groups,
        )
        self.assertNotIn("linux_cpu_artifact", groups)
        self.assertIn(
            "include: "
            "${{ fromJson(needs.changes.outputs.linux_test_groups_json) }}",
            groups,
        )
        self.assertNotIn("- group: protocol", groups)
        self.assertNotIn("- group: skippy-smoke", groups)

    def test_linux_tests_share_one_static_abi_producer(self) -> None:
        producer = job_section(
            self.workflow,
            "linux_static_abi_input",
            "rust_crate_tests",
        )
        crate_tests = job_section(
            self.workflow,
            "rust_crate_tests",
            "linux_test_groups",
        )
        grouped_tests = job_section(
            self.workflow,
            "linux_test_groups",
            "linux_public_mesh_admission",
        )

        self.assertIn("run: scripts/build-llama.sh", producer)
        self.assertIn("name: pr-linux-static-abi-input", producer)
        self.assertIn("mesh-llm-static-abi.tar.gz", producer)
        for consumer in (crate_tests, grouped_tests):
            with self.subTest(consumer=consumer.splitlines()[0].strip()):
                self.assertIn("linux_static_abi_input", consumer)
                self.assertIn("name: pr-linux-static-abi-input", consumer)
                self.assertIn("Restore immutable static ABI input", consumer)
                self.assertNotIn("run: scripts/build-llama.sh", consumer)
                self.assertNotIn("Cache patched llama.cpp ABI build", consumer)

    def test_macos_producers_keep_the_existing_product_route(self) -> None:
        route = (
            "if: ${{ needs.changes.outputs.macos_inference_artifact_required "
            "== 'true' && needs.changes.outputs.docs_only != 'true' }}"
        )

        self.assertIn("needs: changes", self.macos_host)
        self.assertIn(route, self.macos_host)
        self.assertIn("needs: changes", self.macos_runtime)
        self.assertIn(route, self.macos_runtime)

    def test_macos_host_and_runtime_are_independent_producers(self) -> None:
        self.assertIn(
            "uses: ./.github/actions/prepare-host-input",
            self.macos_host,
        )
        self.assertIn("profile: debug", self.macos_host)
        self.assertIn("name: pr-macos-host-input", self.macos_host)
        self.assertNotIn("prepare-native-runtime-input", self.macos_host)
        self.assertNotIn("compose-product-input", self.macos_host)

        self.assertIn(
            "uses: ./.github/actions/prepare-native-runtime-input",
            self.macos_runtime,
        )
        self.assertIn(
            "LLAMA_STAGE_BUILD_DIR: "
            ".deps/llama-build/build-stage-abi-dynamic-metal",
            self.macos_runtime,
        )
        self.assertIn("backend: metal", self.macos_runtime)
        self.assertIn("target: aarch64-apple-darwin", self.macos_runtime)
        self.assertIn(
            "name: pr-macos-metal-runtime-input",
            self.macos_runtime,
        )
        self.assertNotIn("macos_host_input", self.macos_runtime)
        self.assertNotIn("prepare-host-input", self.macos_runtime)
        self.assertNotIn("compose-product-input", self.macos_runtime)

    def test_macos_product_only_composes_immutable_inputs(self) -> None:
        self.assertIn(
            "needs: [changes, macos_host_input, macos_metal_runtime_input]",
            self.macos_product,
        )
        self.assertIn(
            "needs.macos_host_input.result == 'success' "
            "&& needs.macos_metal_runtime_input.result == 'success'",
            self.macos_product,
        )
        self.assertIn("name: pr-macos-host-input", self.macos_product)
        self.assertIn(
            "name: pr-macos-metal-runtime-input",
            self.macos_product,
        )
        self.assertIn(
            "uses: ./.github/actions/compose-product-input",
            self.macos_product,
        )
        self.assertIn("backend: metal", self.macos_product)
        self.assertIn("output_dir: ci-product", self.macos_product)
        self.assertIn(
            "name: ci-macos-inference-binaries",
            self.macos_product,
        )
        self.assertIn(
            "path: ${{ steps.compose.outputs.archive_path }}",
            self.macos_product,
        )
        self.assertNotIn("prepare-host-input", self.macos_product)
        self.assertNotIn("prepare-native-runtime-input", self.macos_product)
        self.assertNotIn("scripts/build-host.sh", self.macos_product)
        self.assertNotIn(
            "scripts/package-native-runtime.sh",
            self.macos_product,
        )
        self.assertNotIn("brew install", self.macos_product)
        self.assertNotIn("Swatinem/rust-cache", self.macos_product)

    def test_macos_swift_gate_and_supported_targets_are_preserved(self) -> None:
        swift = job_section(
            self.workflow,
            "swift_sdk_smoke",
            "macos_unit_tests",
        )
        unit_tests = job_section(
            self.workflow,
            "macos_unit_tests",
            "windows_checks",
        )

        self.assertIn(
            "needs: [changes, macos_cpu_artifact, macos_unit_tests]",
            swift,
        )
        self.assertIn("!cancelled()", swift)
        self.assertNotIn("always()", swift)
        self.assertIn("needs.macos_cpu_artifact.result == 'success'", swift)
        self.assertIn("needs.macos_unit_tests.result == 'success'", swift)
        self.assertIn("needs.macos_unit_tests.result == 'skipped'", swift)
        self.assertIn("artifact_name: ci-macos-inference-binaries", swift)
        self.assertIn("needs: changes", unit_tests)
        self.assertNotIn("macos_cpu_artifact", unit_tests)
        self.assertIn(
            "LLAMA_STAGE_BUILD_DIR: "
            ".deps/llama-build/build-stage-abi-static-metal",
            unit_tests,
        )
        self.assertIn(
            "actions/checkout@"
            "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
            unit_tests,
        )
        self.assertIn(
            "dtolnay/rust-toolchain@"
            "4cda84d5c5c54efe2404f9d843567869ab1699d4",
            unit_tests,
        )
        self.assertIn(
            "Swatinem/rust-cache@"
            "e18b497796c12c097a38f9edb9d0641fb99eee32",
            unit_tests,
        )
        self.assertIn(
            "actions/cache@"
            "caa296126883cff596d87d8935842f9db880ef25",
            unit_tests,
        )
        self.assertNotIn("  macos_targets:", self.workflow)
        self.assertNotIn("Skip unsupported macOS GPU backend", self.workflow)

    def test_windows_pr_keeps_broad_rust_signals_lightweight(self) -> None:
        self.assertIn("needs.changes.outputs.all_rust == 'true'", self.windows_checks)
        self.assertIn("name: Windows lightweight checks", self.windows_checks)
        self.assertIn("cargo check --locked -p mesh-llm --bin mesh-llm", self.windows_checks)
        self.assertNotIn("prepare-windows-host-input", self.windows_checks)
        self.assertNotIn("prepare-native-runtime-input", self.windows_checks)
        self.assertNotIn("compose-product-input", self.windows_checks)

        for producer in (
            self.windows_host,
            self.windows_cpu_runtime,
            self.windows_gpu_runtimes,
        ):
            with self.subTest(producer=producer.splitlines()[0].strip()):
                self.assertNotIn("needs.changes.outputs.all_rust", producer)

    def test_windows_pr_builds_one_debug_host_and_independent_runtimes(self) -> None:
        self.assertIn(
            "uses: ./.github/actions/prepare-windows-host-input",
            self.windows_host,
        )
        self.assertIn("profile: debug", self.windows_host)
        self.assertIn("name: pr-windows-host-input", self.windows_host)
        self.assertNotIn("prepare-native-runtime-input", self.windows_host)
        self.assertNotIn("compose-product-input", self.windows_host)

        self.assertIn(
            "needs.changes.outputs.windows_cpu == 'true'",
            self.windows_cpu_runtime,
        )
        self.assertNotIn(
            "needs.changes.outputs.windows_gpu == 'true'",
            self.windows_cpu_runtime,
        )
        self.assertIn(
            "uses: ./.github/actions/prepare-native-runtime-input",
            self.windows_cpu_runtime,
        )
        self.assertIn("backend: cpu", self.windows_cpu_runtime)
        self.assertIn(
            "name: pr-windows-cpu-runtime-input",
            self.windows_cpu_runtime,
        )
        self.assertNotIn("prepare-windows-host-input", self.windows_cpu_runtime)
        self.assertNotIn("compose-product-input", self.windows_cpu_runtime)

        self.assertIn(
            "needs.changes.outputs.windows_gpu == 'true'",
            self.windows_gpu_runtimes,
        )
        self.assertNotIn(
            "needs.changes.outputs.windows_cpu == 'true'",
            self.windows_gpu_runtimes,
        )
        for backend in ("cuda", "rocm", "vulkan"):
            self.assertIn(f"backend: {backend}", self.windows_gpu_runtimes)
        self.assertIn(
            "uses: ./.github/actions/prepare-native-runtime-input",
            self.windows_gpu_runtimes,
        )
        self.assertIn(
            "name: pr-windows-${{ matrix.backend }}-runtime-input",
            self.windows_gpu_runtimes,
        )
        self.assertNotIn("prepare-windows-host-input", self.windows_gpu_runtimes)
        self.assertNotIn("compose-product-input", self.windows_gpu_runtimes)

    def test_windows_pr_products_only_compose_matching_inputs(self) -> None:
        products = (
            (
                self.windows_cpu_product,
                "pr-windows-cpu-runtime-input",
                "backend: cpu",
            ),
            (
                self.windows_gpu_products,
                "pr-windows-${{ matrix.backend }}-runtime-input",
                "backend: ${{ matrix.backend }}",
            ),
        )

        for product, runtime_artifact, backend in products:
            with self.subTest(product=product.splitlines()[0].strip()):
                self.assertIn("name: pr-windows-host-input", product)
                self.assertIn(f"name: {runtime_artifact}", product)
                self.assertIn(
                    "uses: ./.github/actions/compose-product-input",
                    product,
                )
                self.assertIn(backend, product)
                self.assertIn("binary_name: mesh-llm.exe", product)
                self.assertIn('readiness_smoke: "true"', product)
                self.assertNotIn("prepare-windows-host-input", product)
                self.assertNotIn("prepare-native-runtime-input", product)
                self.assertNotIn("rust-toolchain", product)
                self.assertNotIn("rust-cache", product)
                self.assertNotIn("sccache-action", product)
                self.assertNotIn("cargo ", product)
                self.assertNotIn("build-windows.ps1", product)

        self.assertNotIn("  windows_targets:", self.workflow)


if __name__ == "__main__":
    unittest.main()
