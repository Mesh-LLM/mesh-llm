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


def planned_condition(job_name: str) -> str:
    return (
        "if: ${{ contains("
        "fromJson(needs.changes.outputs.required_jobs_json), "
        f"'{job_name}') }}}}"
    )


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
            "linux_cuda_runtime_input",
        )
        cls.backend_runtimes = {
            "cuda": job_section(
                cls.workflow,
                "linux_cuda_runtime_input",
                "linux_cuda_product",
            ),
            "rocm": job_section(
                cls.workflow,
                "linux_rocm_runtime_input",
                "linux_rocm_product",
            ),
            "vulkan": job_section(
                cls.workflow,
                "linux_vulkan_runtime_input",
                "linux_vulkan_product",
            ),
        }
        cls.backend_products = {
            "cuda": job_section(
                cls.workflow,
                "linux_cuda_product",
                "linux_rocm_runtime_input",
            ),
            "rocm": job_section(
                cls.workflow,
                "linux_rocm_product",
                "linux_vulkan_runtime_input",
            ),
            "vulkan": job_section(
                cls.workflow,
                "linux_vulkan_product",
                "linux_static_abi_input",
            ),
        }
        cls.macos_host = job_section(
            cls.workflow,
            "macos_host_input",
            "macos_metal_runtime_input",
        )
        cls.swift_input = job_section(
            cls.workflow,
            "swift_sdk_input",
            "macos_host_input",
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
            "summary",
        )

    def test_host_profile_covers_every_backend_product_route(self) -> None:
        self.assertIn(planned_condition("linux_host_input"), self.host)
        self.assertIn(
            "needs.changes.outputs.backend_changed == 'true' "
            "|| needs.changes.outputs.benchmarks == 'true'",
            self.host,
        )
        self.assertIn("&& 'release' || 'debug'", self.host)

        for backend, runtime in self.backend_runtimes.items():
            self.assertIn(
                planned_condition(f"linux_{backend}_runtime_input"),
                runtime,
            )

    def test_cpu_runtime_only_runs_for_cpu_product_consumers(self) -> None:
        self.assertIn(
            planned_condition("linux_cpu_runtime_input"),
            self.cpu_runtime,
        )
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

    def test_backend_runtime_inputs_are_independent_producers(self) -> None:
        artifacts = {
            "cuda": "pr-linux-cuda-runtime-input",
            "rocm": "pr-linux-rocm-runtime-input",
            "vulkan": "pr-linux-vulkan-runtime-input",
        }
        expected = {
            "cuda": (
                "sha256:c5b85ef527230f77cf9933ef40bcb44316f9bbcb8fd2ce0651b58acda5143dfd",
                'LLAMA_STAGE_CUDA_ARCHITECTURES: "86"',
            ),
            "rocm": (
                "sha256:6b88ca9371ada2c507d6e36b71f0e0538fee378c6a5e2b39c17249b4b7e5088a",
                "LLAMA_STAGE_AMDGPU_TARGETS: gfx1100",
            ),
            "vulkan": (
                "sha256:ce55fed5c680cd3184b5d4770d9a77c43a702687690906e5753efd2cea27ed80",
                "build-stage-abi-dynamic-vulkan",
            ),
        }

        self.assertNotIn("  linux_targets:", self.workflow)
        for backend, runtime in self.backend_runtimes.items():
            with self.subTest(backend=backend):
                self.assertIn("needs: changes", runtime)
                self.assertIn(
                    "runs-on: ${{ needs.changes.outputs.runner_8 }}",
                    runtime,
                )
                self.assertIn(expected[backend][0], runtime)
                self.assertIn(expected[backend][1], runtime)
                self.assertIn(
                    "uses: ./.github/actions/prepare-native-runtime-input",
                    runtime,
                )
                self.assertIn(f"backend: {backend}", runtime)
                self.assertIn(f"name: {artifacts[backend]}", runtime)
                self.assertIn("runtime-input/*.tar.gz", runtime)
                self.assertIn("runtime-input/*.sha256", runtime)
                self.assertNotIn("linux_host_input", runtime)
                self.assertNotIn("name: pr-linux-host-input", runtime)
                self.assertNotIn("prepare-host-input", runtime)
                self.assertNotIn("compose-product-input", runtime)

        self.assertIn("Cache Vulkan ABI build", self.backend_runtimes["vulkan"])
        self.assertNotIn("Cache Vulkan ABI build", self.backend_runtimes["cuda"])
        self.assertNotIn("Cache Vulkan ABI build", self.backend_runtimes["rocm"])

    def test_backend_products_reuse_exact_immutable_inputs(self) -> None:
        neutral_image = (
            "sha256:8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d"
        )

        self.assertNotIn("pr-linux-release-host-input", self.workflow)
        for backend, product in self.backend_products.items():
            with self.subTest(backend=backend):
                self.assertIn(
                    "needs: [changes, linux_host_input, "
                    f"linux_{backend}_runtime_input]",
                    product,
                )
                self.assertIn(
                    planned_condition(f"linux_{backend}_product"),
                    product,
                )
                self.assertIn(
                    "runs-on: ${{ needs.changes.outputs.runner_4 }}",
                    product,
                )
                self.assertIn(neutral_image, product)
                self.assertIn("name: pr-linux-host-input", product)
                self.assertIn("path: host-input", product)
                self.assertIn(
                    f"name: pr-linux-{backend}-runtime-input",
                    product,
                )
                self.assertIn("path: runtime-input", product)
                self.assertIn(
                    "uses: ./.github/actions/compose-product-input",
                    product,
                )
                self.assertIn(f"backend: {backend}", product)
                self.assertIn("output_dir: product-input", product)
                self.assertIn(f"name: pr-linux-{backend}-product", product)
                self.assertIn(
                    "path: ${{ steps.compose.outputs.archive_path }}",
                    product,
                )
                self.assertNotIn("prepare-native-runtime-input", product)
                self.assertNotIn("configure-sccache-gha", product)
                self.assertNotIn("LLAMA_STAGE_BUILD_DIR", product)
                self.assertNotIn("matrix.", product)

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

        self.assertIn(
            planned_condition("linux_public_mesh_admission"),
            admission,
        )
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

        self.assertIn(
            "uses: ./.github/workflows/static-abi-artifact.yml",
            producer,
        )
        self.assertIn("artifact_name: pr-linux-static-abi-input", producer)
        self.assertIn("runner_size: '8'", producer)
        self.assertNotIn("runs_on:", producer)
        self.assertNotIn("allow_depot_remote_cache:", producer)
        self.assertIn(
            planned_condition("linux_static_abi_input"),
            producer,
        )
        for consumer in (crate_tests, grouped_tests):
            with self.subTest(consumer=consumer.splitlines()[0].strip()):
                self.assertIn("linux_static_abi_input", consumer)
                self.assertIn("name: pr-linux-static-abi-input", consumer)
                self.assertIn("Restore immutable static ABI input", consumer)
                self.assertIn("scripts/restore-static-abi-input.sh", consumer)
                self.assertNotIn("tar -xzf", consumer)
                self.assertNotIn("run: scripts/build-llama.sh", consumer)
                self.assertNotIn("Cache patched llama.cpp ABI build", consumer)

    def test_macos_producers_keep_the_existing_product_route(self) -> None:
        self.assertIn("needs: changes", self.macos_host)
        self.assertIn(
            planned_condition("macos_host_input"),
            self.macos_host,
        )
        self.assertIn("needs: changes", self.macos_runtime)
        self.assertIn(
            planned_condition("macos_metal_runtime_input"),
            self.macos_runtime,
        )

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
            planned_condition("macos_cpu_artifact"),
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

    def test_kotlin_smoke_reuses_parallel_debug_native_sdk_input(self) -> None:
        producer = job_section(
            self.workflow,
            "kotlin_sdk_input",
            "kotlin_sdk_smoke",
        )
        consumer = job_section(
            self.workflow,
            "kotlin_sdk_smoke",
            "swift_sdk_input",
        )

        self.assertIn(
            "needs: [changes, linux_static_abi_input]",
            producer,
        )
        self.assertNotIn("linux_cpu_artifact", producer)
        self.assertIn(
            planned_condition("kotlin_sdk_input"),
            producer,
        )
        self.assertIn(
            "uses: ./.github/workflows/native-sdk-artifact.yml",
            producer,
        )
        self.assertIn("profile: debug", producer)
        self.assertIn(
            "artifact_name: pr-kotlin-native-sdk-input",
            producer,
        )
        self.assertIn(
            "static_abi_artifact_name: pr-linux-static-abi-input",
            producer,
        )
        self.assertIn("runner_size: '8'", producer)
        self.assertNotIn("runs_on:", producer)
        self.assertNotIn("allow_depot_remote_cache:", producer)

        self.assertIn(
            "needs: [changes, linux_cpu_artifact, kotlin_sdk_input]",
            consumer,
        )
        self.assertIn(
            planned_condition("kotlin_sdk_smoke"),
            consumer,
        )
        self.assertIn(
            "kotlin_artifact_name: pr-kotlin-native-sdk-input",
            consumer,
        )
        self.assertIn("kotlin_artifact_profile: debug", consumer)
        self.assertIn(
            "uses: ./.github/workflows/sdk-smoke.yml",
            consumer,
        )

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

        self.assertIn("needs: changes", self.swift_input)
        self.assertIn(
            "uses: ./.github/workflows/swift-sdk-artifact.yml",
            self.swift_input,
        )
        self.assertIn("mode: host-only", self.swift_input)
        self.assertIn("artifact_name: pr-swift-sdk-input", self.swift_input)
        self.assertNotIn("macos_runner:", self.swift_input)
        self.assertIn(
            planned_condition("swift_sdk_input"),
            self.swift_input,
        )
        self.assertNotIn("macos_cpu_artifact", self.swift_input)
        self.assertNotIn("macos_unit_tests", self.swift_input)

        self.assertIn(
            "needs: [changes, macos_cpu_artifact, swift_sdk_input]",
            swift,
        )
        self.assertNotIn("always()", swift)
        self.assertIn(planned_condition("swift_sdk_smoke"), swift)
        self.assertNotIn("macos_unit_tests", swift)
        self.assertIn("artifact_name: ci-macos-inference-binaries", swift)
        self.assertIn("swift_artifact_name: pr-swift-sdk-input", swift)
        self.assertIn("swift_artifact_mode: host-only", swift)
        self.assertNotIn("macos_runner:", swift)
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
        self.assertIn(
            planned_condition("windows_checks"),
            self.windows_checks,
        )
        self.assertIn("name: Windows lightweight checks", self.windows_checks)
        self.assertIn("cargo check --locked -p mesh-llm --bin mesh-llm", self.windows_checks)
        self.assertIn(
            "name: Test Windows log artifact privacy ACL",
            self.windows_checks,
        )
        self.assertIn(
            "github.event_name == 'workflow_dispatch' || "
            "contains(fromJson(needs.changes.outputs.affected_crates), "
            "'mesh-llm-log-store')",
            self.windows_checks,
        )
        self.assertIn(
            "cargo test --locked -p mesh-llm-log-store --lib "
            "windows_artifact_paths_have_current_owner_and_exact_user_only_dacl",
            self.windows_checks,
        )
        self.assertIn(
            "name: Test Windows log SQLite storage ACL",
            self.windows_checks,
        )
        self.assertIn(
            "cargo test --locked -p mesh-llm-log-store --lib "
            "sqlite_root_database_and_sidecars_have_only_current_user_acl",
            self.windows_checks,
        )
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
            planned_condition("windows_cpu_runtime_input"),
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
            planned_condition("windows_gpu_runtime_inputs"),
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
