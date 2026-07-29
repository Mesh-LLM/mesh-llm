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

    def test_windows_packaged_cli_smoke_checks_each_native_command(self) -> None:
        workflow = PR_BUILDS.read_text(encoding="utf-8")
        start = workflow.index("      - name: Composed Windows CLI and client readiness smoke")
        smoke = workflow[start:]

        self.assertIn(".\\target\\release\\mesh-llm.exe --log-format json --version", smoke)
        self.assertIn(".\\target\\release\\mesh-llm.exe --log-format json runtime list", smoke)
        self.assertIn("mesh-llm --version failed with exit code $LASTEXITCODE", smoke)
        self.assertIn("mesh-llm --help failed with exit code $LASTEXITCODE", smoke)
        self.assertIn("mesh-llm runtime list failed with exit code $LASTEXITCODE", smoke)
        self.assertIn("failed with exit code $LASTEXITCODE", smoke)


if __name__ == "__main__":
    unittest.main()
