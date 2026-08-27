import pathlib
import unittest

from scripts.tests.justfile_source import read_justfile_source


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build-development-product.sh"
JUSTFILE = ROOT / "Justfile"


class DevelopmentProductBuildTests(unittest.TestCase):
    def test_builds_dynamic_host_before_exactly_one_runtime(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"$SCRIPT_DIR/build-host.sh" --profile "$PROFILE"', contents)
        self.assertIn('package-native-runtime.sh" "${runtime_args[@]}"', contents)
        self.assertIn('runtime_out="$host_dir/native-runtimes"', contents)
        self.assertNotIn("build-llama.sh", contents)

    def test_linux_default_retains_backend_detection_order(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        cuda = contents.index("BACKEND=cuda")
        rocm = contents.index("BACKEND=rocm")
        vulkan = contents.index("BACKEND=vulkan")
        cpu = contents.index("BACKEND=cpu", vulkan)
        self.assertLess(cuda, rocm)
        self.assertLess(rocm, vulkan)
        self.assertLess(vulkan, cpu)
        self.assertIn("vulkaninfo --summary", contents)
        self.assertIn("pkg-config --exists vulkan", contents)

    def test_build_runtime_empty_backend_defaults_to_cpu(self) -> None:
        # `$$backend` here read the shell PID, not the recipe argument. The
        # behavioral check that this actually defaults (and that an explicit
        # backend survives) lives in test_justfile_release_runtime.py.
        justfile = read_justfile_source(JUSTFILE)
        recipe = justfile[justfile.index('build-runtime backend=""'):]
        self.assertIn('[[ -n "$backend" ]] || backend=cpu', recipe)

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
