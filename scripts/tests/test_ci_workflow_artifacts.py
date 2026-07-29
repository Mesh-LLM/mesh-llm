from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


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

    def test_gpu_products_reuse_the_neutral_host(self) -> None:
        artifacts = {
            "linux_cuda": "ci-linux-cuda-product",
            "linux_rocm": "ci-linux-rocm-product",
            "linux_vulkan": "ci-linux-vulkan-product",
        }

        for job_name, product_artifact in artifacts.items():
            with self.subTest(job=job_name):
                job = job_section(self.workflow, job_name)
                self.assertIn("needs: [changes, linux_host_input]", job)
                self.assertIn("name: ci-linux-host-input", job)
                self.assertIn(
                    "uses: ./.github/actions/prepare-native-runtime-input",
                    job,
                )
                self.assertIn("uses: ./.github/actions/compose-product-input", job)
                self.assertIn(f"name: {product_artifact}", job)
                self.assertNotIn("prepare-host-input", job)
                self.assertNotIn("scripts/build-host.sh", job)

    def test_linux_tests_share_one_static_abi_producer(self) -> None:
        producer = job_section(self.workflow, "linux_static_abi_input")
        crate_tests = job_section(self.workflow, "rust_crate_tests")
        grouped_tests = job_section(self.workflow, "linux_test_groups")

        self.assertIn("run: scripts/build-llama.sh", producer)
        self.assertIn("name: ci-linux-static-abi-input", producer)
        self.assertIn("mesh-llm-static-abi.tar.gz", producer)
        for consumer in (crate_tests, grouped_tests):
            with self.subTest(consumer=consumer.splitlines()[0].strip()):
                self.assertIn("linux_static_abi_input", consumer)
                self.assertIn("name: ci-linux-static-abi-input", consumer)
                self.assertIn("Restore immutable static ABI input", consumer)
                self.assertNotIn("run: scripts/build-llama.sh", consumer)
                self.assertNotIn("Cache patched llama.cpp ABI build", consumer)

    def test_macos_host_and_runtime_are_independent_producers(self) -> None:
        route = (
            "if: ${{ (github.event_name == 'workflow_dispatch' || "
            "needs.changes.outputs.rust == 'true' || "
            "needs.changes.outputs.ui == 'true' || "
            "needs.changes.outputs.benchmarks == 'true') && "
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

    def test_swift_smoke_uses_composed_macos_product(self) -> None:
        swift = job_section(self.workflow, "swift_sdk_smoke")

        self.assertIn(
            "needs: [changes, macos_cpu_artifact, macos_unit_tests]",
            swift,
        )
        self.assertIn("needs.macos_cpu_artifact.result == 'success'", swift)
        self.assertIn("needs.macos_unit_tests.result == 'success'", swift)
        self.assertIn("needs.macos_unit_tests.result == 'skipped'", swift)
        self.assertIn("artifact_name: ci-macos-inference-binaries", swift)
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


if __name__ == "__main__":
    unittest.main()
