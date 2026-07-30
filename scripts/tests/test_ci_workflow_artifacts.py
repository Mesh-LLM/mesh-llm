from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
NODE_ADDON_WORKFLOW = (
    ROOT / ".github" / "workflows" / "node-sdk-addon-artifact.yml"
)


def job_section(workflow: str, job_name: str) -> str:
    marker = f"  {job_name}:\n"
    start = workflow.index(marker)
    next_job = re.search(r"(?m)^  [a-zA-Z0-9_]+:\n", workflow[start + len(marker) :])
    if next_job is None:
        return workflow[start:]
    return workflow[start : start + len(marker) + next_job.start()]


class CiWorkflowArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    def test_release_host_has_one_neutral_producer(self) -> None:
        host = job_section(self.workflow, "linux_host_input")

        self.assertIn("name: Linux immutable release host", host)
        self.assertIn("uses: ./.github/actions/prepare-host-input", host)
        self.assertIn("profile: release", host)
        self.assertIn("name: ci-linux-host-input", host)
        self.assertNotIn("prepare-native-runtime-input", host)
        self.assertNotIn("compose-product-input", host)
        self.assertNotIn("linux_release_host_input:", self.workflow)
        self.assertNotIn("ci-linux-release-host-input", self.workflow)

    def test_arc_runner_contract_is_trusted_main_only(self) -> None:
        arc = job_section(self.workflow, "arc_runner_image_contract")
        pr_workflow = (
            ROOT / ".github" / "workflows" / "pr_builds.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("github.ref == 'refs/heads/main'", arc)
        self.assertIn("runner: mesh-llm-amd64", arc)
        self.assertIn("runner: mesh-llm-arm64", arc)
        self.assertNotIn("arc_runner_image_contract:", pr_workflow)
        self.assertNotIn("runner: mesh-llm-amd64", pr_workflow)
        self.assertNotIn("runner: mesh-llm-arm64", pr_workflow)

    def test_main_uses_event_range_and_centralized_platform_routes(self) -> None:
        changes = job_section(self.workflow, "changes")

        self.assertIn("base_sha: ${{ github.event.before || '' }}", changes)
        self.assertIn("head_sha: ${{ github.sha }}", changes)
        self.assertIn(
            "windows_cpu: ${{ steps.compute.outputs.windows_cpu_build_required }}",
            changes,
        )
        self.assertIn(
            "windows_gpu: ${{ steps.compute.outputs.windows_gpu_build_required }}",
            changes,
        )
        self.assertNotIn("steps.filter.outputs.windows_cpu", changes)
        self.assertNotIn("steps.filter.outputs.windows_gpu", changes)
        swift = job_section(self.workflow, "swift_sdk_smoke")
        self.assertIn("needs.changes.outputs.sdk_smoke_required == 'true'", swift)
        self.assertNotIn("needs.changes.outputs.sdk == 'true'", swift)

    def test_cpu_runtime_is_an_independent_producer(self) -> None:
        runtime = job_section(self.workflow, "linux_cpu_runtime_input")

        self.assertIn("needs: changes", runtime)
        self.assertIn(
            "if: ${{ needs.changes.outputs.docs_only != 'true' }}",
            runtime,
        )
        self.assertIn("uses: ./.github/actions/prepare-native-runtime-input", runtime)
        self.assertIn("backend: cpu", runtime)
        self.assertIn("name: ci-linux-cpu-runtime-input", runtime)
        self.assertNotIn("linux_host_input", runtime)
        self.assertNotIn("prepare-host-input", runtime)
        self.assertNotIn("compose-product-input", runtime)

    def test_cpu_product_only_composes_immutable_inputs(self) -> None:
        product = job_section(self.workflow, "linux_cpu_artifact")

        self.assertIn(
            "needs: [changes, linux_host_input, linux_cpu_runtime_input]",
            product,
        )
        self.assertIn("name: ci-linux-host-input", product)
        self.assertIn("name: ci-linux-cpu-runtime-input", product)
        self.assertIn("uses: ./.github/actions/compose-product-input", product)
        self.assertIn("name: ci-linux-inference-binaries", product)
        self.assertNotIn("prepare-host-input", product)
        self.assertNotIn("prepare-native-runtime-input", product)
        self.assertNotIn("scripts/build-host.sh", product)
        self.assertNotIn("scripts/package-native-runtime.sh", product)
        self.assertNotIn("configure-sccache-gha", product)

    def test_gpu_runtime_inputs_are_independent_producers(self) -> None:
        expected = {
            "cuda": (
                "sha256:c5b85ef527230f77cf9933ef40bcb44316f9bbcb8fd2ce0651b58acda5143dfd",
                'LLAMA_STAGE_CUDA_ARCHITECTURES: "75;80;86;87;89;90"',
            ),
            "rocm": (
                "sha256:0e13e5d2d2c121df265ff6c69be81e468989e09f81d6b7ff049b110cc0bb0d2b",
                "LLAMA_STAGE_AMDGPU_TARGETS: gfx1100",
            ),
            "vulkan": (
                "sha256:ce55fed5c680cd3184b5d4770d9a77c43a702687690906e5753efd2cea27ed80",
                "build-stage-abi-dynamic-vulkan",
            ),
        }

        for backend, (image, backend_env) in expected.items():
            with self.subTest(backend=backend):
                runtime = job_section(
                    self.workflow,
                    f"linux_{backend}_runtime_input",
                )
                self.assertIn("needs: changes", runtime)
                self.assertIn(
                    "if: ${{ needs.changes.outputs.docs_only != 'true' }}",
                    runtime,
                )
                self.assertIn(
                    "runs-on: ${{ needs.changes.outputs.runner_8 }}",
                    runtime,
                )
                self.assertIn(image, runtime)
                self.assertIn(backend_env, runtime)
                self.assertIn(
                    "uses: ./.github/actions/prepare-native-runtime-input",
                    runtime,
                )
                self.assertIn(f"backend: {backend}", runtime)
                self.assertIn(
                    f"name: ci-linux-{backend}-runtime-input",
                    runtime,
                )
                self.assertIn("runtime-input/*.tar.gz", runtime)
                self.assertIn("runtime-input/*.sha256", runtime)
                self.assertNotIn("linux_host_input", runtime)
                self.assertNotIn("name: ci-linux-host-input", runtime)
                self.assertNotIn("prepare-host-input", runtime)
                self.assertNotIn("compose-product-input", runtime)

        for fused_job in ("linux_cuda", "linux_rocm", "linux_vulkan"):
            self.assertNotIn(f"\n  {fused_job}:\n", self.workflow)

    def test_gpu_products_reuse_exact_immutable_inputs(self) -> None:
        neutral_image = (
            "sha256:8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d"
        )

        for backend in ("cuda", "rocm", "vulkan"):
            with self.subTest(backend=backend):
                product = job_section(
                    self.workflow,
                    f"linux_{backend}_product",
                )
                self.assertIn(
                    "needs: [changes, linux_host_input, "
                    f"linux_{backend}_runtime_input]",
                    product,
                )
                self.assertIn(
                    "needs.linux_host_input.result == 'success' "
                    f"&& needs.linux_{backend}_runtime_input.result == 'success'",
                    product,
                )
                self.assertIn(
                    "runs-on: ${{ needs.changes.outputs.runner_4 }}",
                    product,
                )
                self.assertIn(neutral_image, product)
                self.assertIn("name: ci-linux-host-input", product)
                self.assertIn(
                    f"name: ci-linux-{backend}-runtime-input",
                    product,
                )
                self.assertIn(
                    "uses: ./.github/actions/compose-product-input",
                    product,
                )
                self.assertIn(f"backend: {backend}", product)
                self.assertIn(f"name: ci-linux-{backend}-product", product)
                self.assertNotIn("prepare-host-input", product)
                self.assertNotIn("prepare-native-runtime-input", product)
                self.assertNotIn("configure-sccache-gha", product)
                self.assertNotIn("LLAMA_STAGE_BUILD_DIR", product)
                self.assertNotIn("matrix.", product)

    def test_linux_tests_share_one_static_abi_producer(self) -> None:
        producer = job_section(self.workflow, "linux_static_abi_input")
        crate_tests = job_section(self.workflow, "rust_crate_tests")
        grouped_tests = job_section(self.workflow, "linux_test_groups")

        self.assertIn(
            "uses: ./.github/workflows/static-abi-artifact.yml",
            producer,
        )
        self.assertIn("artifact_name: ci-linux-static-abi-input", producer)
        self.assertIn("runner_size: '8'", producer)
        self.assertNotIn("runs_on:", producer)
        self.assertNotIn("allow_depot_remote_cache:", producer)
        self.assertIn(
            "needs.changes.outputs.sdk_smoke_required == 'true'",
            producer,
        )
        for consumer in (crate_tests, grouped_tests):
            with self.subTest(consumer=consumer.splitlines()[0].strip()):
                self.assertIn("linux_static_abi_input", consumer)
                self.assertIn("name: ci-linux-static-abi-input", consumer)
                self.assertIn("Restore immutable static ABI input", consumer)
                self.assertIn("scripts/restore-static-abi-input.sh", consumer)
                self.assertNotIn("tar -xzf", consumer)
                self.assertNotIn("run: scripts/build-llama.sh", consumer)
                self.assertNotIn("Cache patched llama.cpp ABI build", consumer)

    def test_macos_host_and_runtime_are_independent_producers(self) -> None:
        route = (
            "if: ${{ needs.changes.outputs."
            "macos_inference_artifact_required == 'true' && "
            "needs.changes.outputs.docs_only != 'true' }}"
        )
        host = job_section(self.workflow, "macos_host_input")
        runtime = job_section(self.workflow, "macos_metal_runtime_input")

        self.assertIn(route, host)
        self.assertIn("name: macOS immutable release host", host)
        self.assertIn("uses: ./.github/actions/prepare-host-input", host)
        self.assertIn("profile: release", host)
        self.assertIn("name: ci-macos-host-input", host)
        self.assertNotIn("prepare-native-runtime-input", host)
        self.assertNotIn("compose-product-input", host)

        self.assertIn(route, runtime)
        self.assertIn(
            "uses: ./.github/actions/prepare-native-runtime-input",
            runtime,
        )
        self.assertIn("backend: metal", runtime)
        self.assertIn("name: ci-macos-metal-runtime-input", runtime)
        self.assertNotIn("prepare-host-input", runtime)
        self.assertNotIn("compose-product-input", runtime)
        self.assertNotIn("\n  macos:\n", self.workflow)

    def test_macos_product_only_composes_immutable_inputs(self) -> None:
        product = job_section(self.workflow, "macos_cpu_artifact")

        self.assertIn("name: macOS Metal release product", product)
        self.assertIn(
            "needs: [changes, macos_host_input, macos_metal_runtime_input]",
            product,
        )
        self.assertIn("name: ci-macos-host-input", product)
        self.assertIn("name: ci-macos-metal-runtime-input", product)
        self.assertIn("uses: ./.github/actions/compose-product-input", product)
        self.assertIn("name: ci-macos-inference-binaries", product)
        self.assertNotIn("prepare-host-input", product)
        self.assertNotIn("prepare-native-runtime-input", product)
        self.assertNotIn("rust-toolchain", product)
        self.assertNotIn("rust-cache", product)
        self.assertNotIn("brew install", product)
        self.assertNotIn("cargo ", product)

    def test_new_macos_jobs_pin_their_external_actions(self) -> None:
        host = job_section(self.workflow, "macos_host_input")
        runtime = job_section(self.workflow, "macos_metal_runtime_input")
        product = job_section(self.workflow, "macos_cpu_artifact")
        unit_tests = job_section(self.workflow, "macos_unit_tests")
        checkout = (
            "actions/checkout@"
            "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
        )

        for job in (host, runtime, product, unit_tests):
            with self.subTest(job=job.splitlines()[0].strip()):
                self.assertIn(checkout, job)

        self.assertIn(
            "pnpm/action-setup@"
            "b906affcce14559ad1aafd4ab0e942779e9f58b1",
            host,
        )
        self.assertIn(
            "actions/setup-node@"
            "a0853c24544627f65ddf259abe73b1d18a591444",
            host,
        )
        rust_toolchain = (
            "dtolnay/rust-toolchain@"
            "4cda84d5c5c54efe2404f9d843567869ab1699d4"
        )
        rust_cache = (
            "Swatinem/rust-cache@"
            "e18b497796c12c097a38f9edb9d0641fb99eee32"
        )
        actions_cache = "caa296126883cff596d87d8935842f9db880ef25"
        for job in (host, unit_tests):
            self.assertIn(rust_toolchain, job)
            self.assertIn(rust_cache, job)
            self.assertIn(actions_cache, job)

    def test_macos_unit_tests_keep_static_abi_separate(self) -> None:
        unit_tests = job_section(self.workflow, "macos_unit_tests")

        self.assertIn(
            "LLAMA_STAGE_BUILD_DIR: "
            ".deps/llama-build/build-stage-abi-static-metal",
            unit_tests,
        )
        self.assertIn("name: Cache static Metal ABI build", unit_tests)
        self.assertIn("run: scripts/build-llama.sh", unit_tests)
        self.assertIn("cargo test -p \"$c\" --lib", unit_tests)
        self.assertNotIn("prepare-host-input", unit_tests)
        self.assertNotIn("prepare-native-runtime-input", unit_tests)
        self.assertNotIn("compose-product-input", unit_tests)

    def test_windows_main_reuses_one_release_host_for_all_products(self) -> None:
        host = job_section(self.workflow, "windows_host_input")

        self.assertIn("needs.changes.outputs.rust == 'true'", host)
        self.assertIn("needs.changes.outputs.windows_cpu == 'true'", host)
        self.assertIn("needs.changes.outputs.windows_gpu == 'true'", host)
        self.assertIn(
            "uses: ./.github/actions/prepare-windows-host-input",
            host,
        )
        self.assertIn("profile: release", host)
        self.assertIn("name: ci-windows-host-input", host)
        self.assertNotIn("prepare-native-runtime-input", host)
        self.assertNotIn("compose-product-input", host)
        self.assertNotIn("\n  windows_cpu:\n", self.workflow)
        self.assertNotIn("\n  windows_gpu:\n", self.workflow)

    def test_windows_main_runtime_inputs_are_independent_producers(self) -> None:
        cpu = job_section(self.workflow, "windows_cpu_runtime_input")
        gpu = job_section(self.workflow, "windows_gpu_runtime_inputs")

        self.assertIn("needs.changes.outputs.rust == 'true'", cpu)
        self.assertIn("needs.changes.outputs.windows_cpu == 'true'", cpu)
        self.assertNotIn("needs.changes.outputs.windows_gpu == 'true'", cpu)
        self.assertIn(
            "uses: ./.github/actions/prepare-native-runtime-input",
            cpu,
        )
        self.assertIn("backend: cpu", cpu)
        self.assertIn("name: ci-windows-cpu-runtime-input", cpu)

        self.assertIn("needs.changes.outputs.windows_gpu == 'true'", gpu)
        self.assertNotIn("needs.changes.outputs.rust == 'true'", gpu)
        self.assertNotIn("needs.changes.outputs.windows_cpu == 'true'", gpu)
        for backend in ("cuda", "rocm", "vulkan"):
            self.assertIn(f"backend: {backend}", gpu)
        self.assertIn(
            "uses: ./.github/actions/prepare-native-runtime-input",
            gpu,
        )
        self.assertIn(
            "name: ci-windows-${{ matrix.backend }}-runtime-input",
            gpu,
        )

        for producer in (cpu, gpu):
            with self.subTest(producer=producer.splitlines()[0].strip()):
                self.assertNotIn("prepare-windows-host-input", producer)
                self.assertNotIn("compose-product-input", producer)

    def test_windows_main_products_are_composition_only(self) -> None:
        cpu = job_section(self.workflow, "windows_cpu_product")
        gpu = job_section(self.workflow, "windows_gpu_products")
        products = (
            (
                cpu,
                "ci-windows-cpu-runtime-input",
                "backend: cpu",
            ),
            (
                gpu,
                "ci-windows-${{ matrix.backend }}-runtime-input",
                "backend: ${{ matrix.backend }}",
            ),
        )

        self.assertIn("needs.changes.outputs.rust == 'true'", cpu)
        self.assertIn("needs.changes.outputs.windows_cpu == 'true'", cpu)
        self.assertIn("needs.changes.outputs.windows_gpu == 'true'", gpu)
        self.assertNotIn("needs.changes.outputs.rust == 'true'", gpu)

        for product, runtime_artifact, backend in products:
            with self.subTest(product=product.splitlines()[0].strip()):
                self.assertIn("name: ci-windows-host-input", product)
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

    def test_windows_node_checks_remain_separate_from_product_builds(self) -> None:
        checks = job_section(self.workflow, "windows_node_checks")

        self.assertIn("name: Windows Node SDK checks", checks)
        self.assertIn("cargo check --locked -p mesh-llm-nodejs", checks)
        self.assertNotIn("prepare-windows-host-input", checks)
        self.assertNotIn("prepare-native-runtime-input", checks)
        self.assertNotIn("compose-product-input", checks)

    def test_kotlin_smoke_reuses_parallel_release_native_sdk_input(self) -> None:
        producer = job_section(self.workflow, "kotlin_sdk_input")
        consumer = job_section(self.workflow, "kotlin_sdk_smoke")

        self.assertIn(
            "needs: [changes, linux_static_abi_input]",
            producer,
        )
        self.assertNotIn("linux_cpu_artifact", producer)
        self.assertIn(
            "needs.linux_static_abi_input.result == 'success'",
            producer,
        )
        self.assertIn(
            "needs.changes.outputs.sdk_smoke_required == 'true'",
            producer,
        )
        self.assertIn(
            "uses: ./.github/workflows/native-sdk-artifact.yml",
            producer,
        )
        self.assertIn("profile: release", producer)
        self.assertIn(
            "artifact_name: ci-kotlin-native-sdk-input",
            producer,
        )
        self.assertIn(
            "static_abi_artifact_name: ci-linux-static-abi-input",
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
            "needs.kotlin_sdk_input.result == 'success'",
            consumer,
        )
        self.assertIn(
            "kotlin_artifact_name: ci-kotlin-native-sdk-input",
            consumer,
        )
        self.assertIn("kotlin_artifact_profile: release", consumer)
        self.assertIn(
            "uses: ./.github/workflows/sdk-smoke.yml",
            consumer,
        )

    def test_swift_smoke_uses_composed_macos_product(self) -> None:
        producer = job_section(self.workflow, "swift_sdk_input")
        swift = job_section(self.workflow, "swift_sdk_smoke")

        self.assertIn("needs: changes", producer)
        self.assertIn(
            "uses: ./.github/workflows/swift-sdk-artifact.yml",
            producer,
        )
        self.assertIn("mode: full", producer)
        self.assertIn("artifact_name: ci-swift-sdk-input", producer)
        self.assertIn("timeout_minutes: 180", producer)
        self.assertNotIn("macos_runner:", producer)
        self.assertNotIn("macos_cpu_artifact", producer)
        self.assertNotIn("macos_unit_tests", producer)

        self.assertIn(
            "needs: [changes, macos_cpu_artifact, swift_sdk_input]",
            swift,
        )
        self.assertNotIn("always()", swift)
        self.assertIn("needs.macos_cpu_artifact.result == 'success'", swift)
        self.assertIn("needs.swift_sdk_input.result == 'success'", swift)
        self.assertNotIn("macos_unit_tests", swift)
        self.assertIn("artifact_name: ci-macos-inference-binaries", swift)
        self.assertIn("swift_artifact_name: ci-swift-sdk-input", swift)
        self.assertIn("swift_artifact_mode: full", swift)
        self.assertNotIn("macos_runner:", swift)
        self.assertIn("staged_binary_path: target/release/mesh-llm", swift)

    def test_main_runner_policy_is_selected_once(self) -> None:
        changes = job_section(self.workflow, "changes")

        self.assertIn("runs-on: ubuntu-24.04", changes)
        self.assertIn("uses: ./.github/actions/select-ci-runners", changes)
        self.assertIn(
            "depot_main_enabled: ${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            changes,
        )
        self.assertIn("ref: ${{ github.ref }}", changes)
        self.assertIn(
            "allow_depot_remote_cache: "
            "${{ steps.runners.outputs.allow_depot_remote_cache }}",
            changes,
        )
        self.assertNotIn(
            "(vars.DEPOT_RUNNERS_ENABLED == 'true' || "
            "inputs.use_depot == true)",
            self.workflow,
        )

    def test_linux_product_consumers_stage_the_release_profile(self) -> None:
        linux_consumers = self.workflow[: self.workflow.index("  swift_sdk_smoke:")]

        self.assertNotIn("target/debug/mesh-llm", linux_consumers)
        self.assertIn("target/release/mesh-llm", linux_consumers)


class ReleaseNodeAddonArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        cls.producer = NODE_ADDON_WORKFLOW.read_text(encoding="utf-8")

    def test_release_uses_one_reusable_producer_for_all_node_targets(
        self,
    ) -> None:
        producer = job_section(self.release, "build_node_sdk_addon")

        self.assertIn(
            "uses: ./.github/workflows/node-sdk-addon-artifact.yml",
            producer,
        )
        self.assertIn(
            "artifact_name: release-node-sdk-addon-${{ matrix.target }}",
            producer,
        )
        for target in (
            "darwin-arm64",
            "darwin-x64",
            "linux-arm64",
            "linux-x64",
            "win32-x64",
        ):
            with self.subTest(target=target):
                self.assertEqual(producer.count(f"target: {target}"), 1)
        self.assertNotIn("npm run build:native", producer)
        self.assertNotIn("actions/upload-artifact", producer)

    def test_node_addon_producer_is_non_publishing_and_checksummed(
        self,
    ) -> None:
        self.assertIn("on:\n  workflow_call:", self.producer)
        self.assertIn(
            "mesh-llm-node-sdk-addon-$expected_version-$NODE_SDK_TARGET.tar.gz",
            self.producer,
        )
        self.assertIn(
            '"schema": "mesh-llm-node-sdk-addon-v1"',
            self.producer,
        )
        self.assertIn("currentMeshVersion()", self.producer)
        self.assertIn("npm pack", self.producer)
        self.assertIn("$output_root/$archive.sha256", self.producer)
        self.assertIn(
            "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:",
            self.producer,
        )
        self.assertIn(
            "darwin-arm64|darwin-x64|linux-arm64|linux-x64|win32-x64",
            job_section(self.producer, "validate_inputs"),
        )
        for job_name in ("linux_addon", "macos_addon", "windows_addon"):
            with self.subTest(job=job_name):
                self.assertIn(
                    "needs: validate_inputs",
                    job_section(self.producer, job_name),
                )
        self.assertNotIn("softprops/action-gh-release", self.producer)
        self.assertNotIn("npm publish", self.producer)
        self.assertNotIn("contents: write", self.producer)

    def test_release_publish_requires_and_uploads_node_addons(self) -> None:
        publish = job_section(self.release, "publish")

        self.assertIn("- build_node_sdk_addon", publish)
        self.assertIn(
            "needs.build_node_sdk_addon.result == 'success'",
            publish,
        )
        self.assertIn("pattern: release-*", publish)
        self.assertIn("files: release-artifacts/*", publish)


if __name__ == "__main__":
    unittest.main()
