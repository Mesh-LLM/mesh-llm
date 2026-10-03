import pathlib
import unittest

from scripts.tests.justfile_source import read_justfile_source


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "mesh" / "scripts" / "build-development-product.sh"
SKIPPY_SCRIPT = ROOT / "skippy" / "scripts" / "build-development-product.sh"
WINDOWS_SCRIPT = ROOT / "mesh" / "scripts" / "build-windows.ps1"
WINDOWS_FORWARDER = ROOT / "scripts" / "build-windows.ps1"
JUSTFILE = ROOT / "Justfile"


class DevelopmentProductBuildTests(unittest.TestCase):
    def test_builds_complete_skippy_product_before_dynamic_mesh_host(self) -> None:
        mesh = SCRIPT.read_text(encoding="utf-8")
        skippy = SKIPPY_SCRIPT.read_text(encoding="utf-8")
        self.assertLess(
            mesh.index('just skippy "$BACKEND" "$CUDA_ARCH" "$ROCM_ARCH"'),
            mesh.index('just mesh "$PROFILE"'),
        )
        self.assertLess(
            skippy.index('package-native-runtime.sh" "${runtime_args[@]}"'),
            skippy.index('just skippy-cli-build'),
        )
        self.assertIn('runtime_out="$host_dir/native-runtimes"', mesh)
        self.assertIn('Skippy CLI:     $host_dir/skippy', mesh)
        self.assertNotIn("build-llama.sh", mesh)

    def test_linux_default_retains_backend_detection_order(self) -> None:
        contents = SKIPPY_SCRIPT.read_text(encoding="utf-8")
        cuda = contents.index("BACKEND=cuda")
        rocm = contents.index("BACKEND=rocm")
        vulkan = contents.index("BACKEND=vulkan")
        cpu = contents.index("BACKEND=cpu", vulkan)
        self.assertLess(cuda, rocm)
        self.assertLess(rocm, vulkan)
        self.assertLess(vulkan, cpu)
        self.assertIn("vulkaninfo --summary", contents)
        self.assertIn("pkg-config --exists vulkan", contents)

    def test_native_only_recipe_defaults_to_the_platform_backend(self) -> None:
        justfile = read_justfile_source(JUSTFILE)
        recipe = justfile[justfile.index('release-runtime-build backend=""'):]
        self.assertIn('selected_backend=metal; else selected_backend=cpu', recipe)

    def test_windows_builds_standalone_skippy_before_mesh(self) -> None:
        contents = WINDOWS_SCRIPT.read_text(encoding="utf-8")
        runtime = contents.index('"--out", $runtimeOut')
        skippy = contents.index('Invoke-NativeCommand "cargo" $skippyArgs')
        mesh = contents.index('Write-Host "Building mesh-llm..."')
        self.assertLess(runtime, skippy)
        self.assertLess(skippy, mesh)
        self.assertLess(contents.index('if ($SkippyOnly)'), mesh)
        self.assertIn("[switch]$SkippyOnly", WINDOWS_FORWARDER.read_text(encoding="utf-8"))

    def test_preserves_documented_named_just_arguments(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'BACKEND="$(normalize_recipe_argument "$BACKEND" backend)"',
            contents,
        )
        self.assertIn(
            'CUDA_ARCH="$(normalize_recipe_argument "$CUDA_ARCH" cuda_arch cuda-arch)"',
            contents,
        )
        self.assertIn(
            'ROCM_ARCH="$(normalize_recipe_argument "$ROCM_ARCH" rocm_arch rocm-arch amd_arch amd-arch)"',
            contents,
        )
