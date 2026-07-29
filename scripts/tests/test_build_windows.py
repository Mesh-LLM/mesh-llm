from __future__ import annotations

from pathlib import Path
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "scripts" / "build-windows.ps1"
PR_BUILDS: Final = ROOT / ".github" / "workflows" / "pr_builds.yml"


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
        self.assertIn('Join-Path (Join-Path (Join-Path $repoRoot "target") $profileDir) "native-runtimes"', script)
        self.assertNotIn('"--out", (Join-Path $repoRoot', package_call)

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
