from __future__ import annotations

from pathlib import Path
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "scripts" / "build-windows.ps1"
PR_BUILDS: Final = ROOT / ".github" / "workflows" / "pr_builds.yml"
CI_WORKFLOW: Final = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW: Final = ROOT / ".github" / "workflows" / "release.yml"
WINDOWS_WARM_CACHES: Final = (
    ROOT / ".github" / "workflows" / "windows-warm-caches.yml"
)
CACHE_SAVE_VERIFIER: Final = (
    ROOT
    / ".github"
    / "actions"
    / "save-and-verify-actions-cache"
    / "action.yml"
)


class BuildWindowsScriptTests(unittest.TestCase):
    def test_windows_release_recipes_request_dynamic_hosts(self) -> None:
        justfile = (ROOT / "Justfile").read_text(encoding="utf-8")
        for recipe in (
            "release-build-windows:",
            "release-build-cuda-windows",
            "release-build-rocm-windows",
            "release-build-vulkan-windows:",
        ):
            start = justfile.index(recipe)
            end = justfile.find("\n\n", start)
            self.assertIn("-DynamicHost", justfile[start:end])

    def test_native_runtime_package_out_path_is_git_bash_safe(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        package_call = script[script.index('Invoke-NativeCommand "bash" @(') :]
        package_call = package_call[: package_call.index("\n    )")]

        self.assertIn('"--out", $runtimeOut', package_call)
        self.assertIn(
            '$runtimeOut = "target/$profileDir/native-runtimes"',
            script,
        )
        self.assertNotIn(
            'Join-Path (Join-Path (Join-Path $repoRoot "target")',
            script,
        )
        self.assertNotIn('"--out", (Join-Path $repoRoot', package_call)

    def test_cache_warmers_use_the_canonical_runtime_action(self) -> None:
        justfile = (ROOT / "Justfile").read_text(encoding="utf-8")
        script = SCRIPT.read_text(encoding="utf-8")
        workflow = WINDOWS_WARM_CACHES.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count(
                "uses: ./.github/actions/prepare-native-runtime-input",
            ),
            2,
        )
        self.assertEqual(workflow.count(".mesh-llm-build-stamp"), 2)
        self.assertNotIn("run: just ", workflow)
        self.assertNotIn("build-windows.ps1", workflow)
        self.assertNotIn("release-runtime-abi-", justfile)
        self.assertNotIn("[switch]$AbiOnly", script)

    def test_every_windows_runtime_graph_consumes_the_shared_abi_cache(
        self,
    ) -> None:
        workflows = {
            "pr": PR_BUILDS,
            "main": CI_WORKFLOW,
            "release": RELEASE_WORKFLOW,
            "warmer": WINDOWS_WARM_CACHES,
        }

        for name, path in workflows.items():
            with self.subTest(workflow=name):
                workflow = path.read_text(encoding="utf-8")
                self.assertEqual(
                    workflow.count(
                        "uses: ./.github/actions/restore-windows-abi-cache",
                    ),
                    2,
                )
                self.assertEqual(
                    workflow.count(
                        "name: Resolve Windows native toolchain epoch",
                    ),
                    2,
                )
                self.assertEqual(
                    workflow.count(
                        "toolchain_epoch: "
                        "${{ steps.native_toolchain.outputs.epoch }}",
                    ),
                    2,
                )

        warmer = WINDOWS_WARM_CACHES.read_text(encoding="utf-8")
        self.assertIn(
            "'.github/actions/resolve-native-toolchain-epoch/action.yml'",
            warmer,
        )
        self.assertEqual(
            warmer.count("scripts/verify-static-abi-build-stamp.py"),
            2,
        )
        self.assertEqual(
            warmer.count(
                "--toolchain-epoch "
                "$env:MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
            ),
            2,
        )
        for library_group in (
            '@("llama.dll", "libllama.dll")',
            '@("llama-common.dll", "libllama-common.dll")',
            '@("mtmd.dll", "libmtmd.dll")',
        ):
            with self.subTest(library_group=library_group):
                self.assertEqual(warmer.count(library_group), 2)
        self.assertEqual(
            warmer.count("Save and verify Windows "),
            2,
        )
        self.assertEqual(
            warmer.count(
                "uses: ./.github/actions/save-and-verify-actions-cache",
            ),
            2,
        )
        self.assertEqual(
            warmer.count(
                "cache-key: "
                "${{ steps.llama_cache.outputs.cache-primary-key }}",
            ),
            2,
        )
        self.assertEqual(
            warmer.count(
                "path: ${{ steps.llama_cache.outputs.cache-path }}",
            ),
            2,
        )
        self.assertNotIn(
            "path: ${{ env.LLAMA_STAGE_BUILD_DIR }}",
            warmer,
        )
        self.assertEqual(
            warmer.count("cache-ref: ${{ github.ref }}"),
            2,
        )
        verifier = CACHE_SAVE_VERIFIER.read_text(encoding="utf-8")
        self.assertEqual(
            verifier.count("Snapshot existing exact cache entries"),
            1,
        )
        self.assertIn("!existingIds.has(String(candidate.id))", verifier)
        self.assertIn("candidate.size_in_bytes > 0", verifier)
        self.assertIn("name: Verify current cache version lookup", verifier)
        self.assertIn("lookup-only: true", verifier)
        self.assertIn("fail-on-cache-miss: true", verifier)
        self.assertEqual(
            verifier.count(
                "uses: actions/cache/restore@"
                "caa296126883cff596d87d8935842f9db880ef25",
            ),
            1,
        )
        self.assertEqual(
            verifier.count("attempt <= 12"),
            1,
        )
        self.assertIn(
            "'.github/actions/save-and-verify-actions-cache/action.yml'",
            warmer,
        )

    def test_windows_abi_caches_live_outside_the_llama_worktree(self) -> None:
        workflows = {
            "pr": PR_BUILDS,
            "main": CI_WORKFLOW,
            "release": RELEASE_WORKFLOW,
            "warmer": WINDOWS_WARM_CACHES,
        }
        unsafe_cpu = (
            "LLAMA_STAGE_BUILD_DIR: "
            ".deps/llama.cpp/build-stage-abi-cpu"
        )
        unsafe_matrix = (
            "LLAMA_STAGE_BUILD_DIR: "
            ".deps/llama.cpp/build-stage-abi-${{ matrix.backend }}"
        )
        safe_prefix = (
            "LLAMA_STAGE_BUILD_DIR: "
            ".deps/llama-build/windows/build-stage-abi-"
        )

        for name, path in workflows.items():
            with self.subTest(workflow=name):
                workflow = path.read_text(encoding="utf-8")
                self.assertEqual(workflow.count(safe_prefix), 2)
                self.assertNotIn(unsafe_cpu, workflow)
                self.assertNotIn(unsafe_matrix, workflow)

        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "$buildDir = if ($env:LLAMA_STAGE_BUILD_DIR)",
            script,
        )
        self.assertIn(
            'Join-Path $repoRoot ".deps\\llama-build"',
            script,
        )

    def test_release_windows_gpu_setup_matches_pr_and_main(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("  build_native_runtime_windows_gpu:")
        end = workflow.index("\n  publish:", start)
        windows_gpu = workflow[start:end]

        self.assertIn("uses: Jimver/cuda-toolkit@", windows_gpu)
        self.assertIn(
            "uses: jakoch/install-vulkan-sdk-action@",
            windows_gpu,
        )
        self.assertIn(
            "uses: ./.github/actions/setup-windows-rocm-sdk",
            windows_gpu,
        )
        self.assertNotIn("install-windows-sdk.ps1", windows_gpu)

    def test_all_normal_profiles_build_a_dynamic_host_and_adjacent_runtime(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"-DBUILD_SHARED_LIBS=ON"', script)
        self.assertNotIn("$buildProfile -eq 'release' -or $DynamicHost", script)
        self.assertNotIn('if ($buildProfile -eq "release" -or $DynamicHost)', script)
        self.assertIn(
            '$cargoFeatureArgs = @("--no-default-features", "--features", "web-ui,dynamic-native-runtime")',
            script,
        )
        self.assertNotIn('"dist/native-runtimes"', script)
        self.assertIn(
            "-DynamicHost is retained as a compatibility switch; Windows hosts are always dynamic.",
            script,
        )

    def test_host_only_build_honors_debug_and_release_profiles(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        start = script.index("if ($HostOnly) {")
        end = script.index("\nswitch ($backendName)", start)
        host_only = script[start:end]

        self.assertIn('$hostArgs = @("build")', host_only)
        self.assertIn('if ($buildProfile -eq "release")', host_only)
        self.assertIn('$hostArgs += "--release"', host_only)
        self.assertIn('$hostOutputProfile = "debug"', host_only)
        self.assertIn('$hostOutputProfile = "release"', host_only)
        self.assertNotIn(
            '@("build", "--release", "--locked"',
            host_only,
        )
        self.assertIn("\n    return\n", host_only)
        self.assertNotIn("exit 0", host_only)

    def test_windows_products_use_the_shared_composition_and_smoke_contract(
        self,
    ) -> None:
        workflow = PR_BUILDS.read_text(encoding="utf-8")
        cpu_start = workflow.index("  windows_cpu_product:")
        gpu_start = workflow.index("  windows_gpu_products:", cpu_start)
        products = (workflow[cpu_start:gpu_start], workflow[gpu_start:])

        for product in products:
            with self.subTest(job=product.splitlines()[0].strip()):
                self.assertIn(
                    "uses: ./.github/actions/compose-product-input",
                    product,
                )
                self.assertIn("binary_name: mesh-llm.exe", product)
                self.assertIn('readiness_smoke: "true"', product)
                self.assertNotIn("cargo ", product)
                self.assertNotIn("build-windows.ps1", product)


if __name__ == "__main__":
    unittest.main()
