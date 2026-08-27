from pathlib import Path
import unittest

from scripts.tests.justfile_source import read_justfile_source


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build-windows.ps1"
RUNTIME = ROOT / ".github/workflows/ci-windows-runtime-slice.yml"
PRODUCT = ROOT / ".github/workflows/ci-windows-product-slice.yml"
HOST_INPUT = ROOT / ".github/actions/prepare-windows-host-input/action.yml"


class BuildWindowsScriptTests(unittest.TestCase):
    def test_windows_host_ui_fallback_requires_index_entry_point(self):
        action = HOST_INPUT.read_text()
        self.assertIn('$uiEntryPoint = Join-Path $uiDist "index.html"', action)
        self.assertIn(
            "Test-Path -LiteralPath $uiEntryPoint -PathType Leaf",
            action,
        )
        self.assertNotIn("Get-ChildItem -LiteralPath $uiDist", action)

    def test_windows_release_recipes_request_dynamic_hosts(self):
        justfile = read_justfile_source(ROOT / "Justfile")
        for recipe in (
            "release-build-windows:",
            "release-build-cuda-windows",
            "release-build-rocm-windows",
            "release-build-vulkan-windows:",
        ):
            start = justfile.index(recipe)
            end = justfile.find("\n\n", start)
            self.assertIn("-DynamicHost", justfile[start:end])

    def test_windows_script_keeps_dynamic_host_and_runtime_separation(self):
        script = SCRIPT.read_text()
        self.assertIn('"-DBUILD_SHARED_LIBS=ON"', script)
        self.assertIn('dynamic-native-runtime', script)
        self.assertNotIn("[switch]$AbiOnly", script)

    def test_windows_runtime_slice_uses_verified_cache_and_composer(self):
        runtime = RUNTIME.read_text()
        product = PRODUCT.read_text()
        self.assertIn("restore-windows-abi-cache", runtime)
        self.assertIn("compose-product-input", product)
        self.assertIn("binary_name: mesh-llm.exe", product)
        self.assertIn('readiness_smoke: "true"', product)
        self.assertNotIn("build-windows.ps1", product)


if __name__ == "__main__":
    unittest.main()
