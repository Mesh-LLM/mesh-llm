import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "validate-release-native-runtime-matrix.py"
)


def load_validator():
    spec = importlib.util.spec_from_file_location("release_matrix_validator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseNativeRuntimeMatrixTests(unittest.TestCase):
    def test_binary_asset_inference_requires_matching_runtime_entries(self):
        validator = load_validator()
        assets = [
            "mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu.tar.gz",
            "mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu-cuda-13.tar.gz",
            "meshllm-native-runtime-linux-x86_64-cpu.tar.gz",
        ]
        manifest = {
            "artifacts": [
                {
                    "id": "meshllm-native-runtime-linux-x86_64-cpu",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {"kind": "cpu"},
                }
            ]
        }

        violations = validator.find_matrix_violations(assets, manifest)

        self.assertEqual(
            violations,
            [
                "missing native runtime for binary target linux/aarch64/cpu",
                "missing native runtime for binary target linux/aarch64/cuda13",
            ],
        )

    def test_matching_linux_aarch64_cpu_and_cuda_entries_pass(self):
        validator = load_validator()
        assets = [
            "mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu.tar.gz",
            "mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu-cuda-13.tar.gz",
        ]
        manifest = {
            "artifacts": [
                {
                    "id": "meshllm-native-runtime-linux-aarch64-cpu",
                    "platform": {"os": "linux", "arch": "aarch64"},
                    "backend": {"kind": "cpu"},
                },
                {
                    "id": "meshllm-native-runtime-linux-aarch64-cuda13",
                    "platform": {"os": "linux", "arch": "aarch64"},
                    "backend": {
                        "kind": "cuda",
                        "cuda": {"toolkit_major": 13, "gpu_arches": []},
                    },
                },
            ]
        }

        self.assertEqual(validator.find_matrix_violations(assets, manifest), [])

    def test_explicit_release_native_targets_cover_all_release_bundles(self):
        validator = load_validator()
        assets = [
            "mesh-llm-v0.72.0-rc5-x86_64-unknown-linux-gnu.tar.gz",
            "mesh-llm-v0.72.0-rc5-x86_64-unknown-linux-gnu-cuda-12.tar.gz",
            "mesh-llm-v0.72.0-rc5-x86_64-unknown-linux-gnu-cuda-13.tar.gz",
            "mesh-llm-v0.72.0-rc5-x86_64-unknown-linux-gnu-rocm.tar.gz",
            "mesh-llm-v0.72.0-rc5-x86_64-unknown-linux-gnu-vulkan.tar.gz",
            "mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu.tar.gz",
            "mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu-cuda-12.tar.gz",
            "mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu-cuda-13.tar.gz",
            "mesh-llm-v0.72.0-rc5-x86_64-pc-windows-msvc.zip",
            "mesh-llm-v0.72.0-rc5-x86_64-pc-windows-msvc-cuda.zip",
            "mesh-llm-v0.72.0-rc5-x86_64-pc-windows-msvc-rocm.zip",
            "mesh-llm-v0.72.0-rc5-x86_64-pc-windows-msvc-vulkan.zip",
        ]
        manifest = {
            "artifacts": [
                {
                    "id": "meshllm-native-runtime-darwin-aarch64-metal",
                    "platform": {"os": "macos", "arch": "aarch64"},
                    "backend": {"kind": "metal"},
                },
                {
                    "id": "meshllm-native-runtime-linux-x86_64-cpu",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {"kind": "cpu"},
                },
                {
                    "id": "meshllm-native-runtime-linux-aarch64-cpu",
                    "platform": {"os": "linux", "arch": "aarch64"},
                    "backend": {"kind": "cpu"},
                },
                {
                    "id": "meshllm-native-runtime-linux-aarch64-cuda12",
                    "platform": {"os": "linux", "arch": "aarch64"},
                    "backend": {
                        "kind": "cuda",
                        "cuda": {"toolkit_major": 12, "gpu_arches": []},
                    },
                },
                {
                    "id": "meshllm-native-runtime-linux-aarch64-cuda13",
                    "platform": {"os": "linux", "arch": "aarch64"},
                    "backend": {
                        "kind": "cuda",
                        "cuda": {"toolkit_major": 13, "gpu_arches": []},
                    },
                },
                {
                    "id": "meshllm-native-runtime-linux-x86_64-cuda12",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {
                        "kind": "cuda",
                        "cuda": {"toolkit_major": 12, "gpu_arches": []},
                    },
                },
                {
                    "id": "meshllm-native-runtime-linux-x86_64-cuda13",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {
                        "kind": "cuda",
                        "cuda": {"toolkit_major": 13, "gpu_arches": []},
                    },
                },
                {
                    "id": "meshllm-native-runtime-linux-x86_64-rocm",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {"kind": "rocm"},
                },
                {
                    "id": "meshllm-native-runtime-linux-x86_64-vulkan",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {"kind": "vulkan"},
                },
                {
                    "id": "meshllm-native-runtime-windows-x86_64-cpu",
                    "platform": {"os": "windows", "arch": "x86_64"},
                    "backend": {"kind": "cpu"},
                },
                {
                    "id": "meshllm-native-runtime-windows-x86_64-cuda12",
                    "platform": {"os": "windows", "arch": "x86_64"},
                    "backend": {
                        "kind": "cuda",
                        "cuda": {"toolkit_major": 12, "gpu_arches": []},
                    },
                },
                {
                    "id": "meshllm-native-runtime-windows-x86_64-rocm",
                    "platform": {"os": "windows", "arch": "x86_64"},
                    "backend": {"kind": "rocm"},
                },
                {
                    "id": "meshllm-native-runtime-windows-x86_64-vulkan",
                    "platform": {"os": "windows", "arch": "x86_64"},
                    "backend": {"kind": "vulkan"},
                },
            ]
        }
        required_targets = {
            validator.target_from_label("macos/aarch64/metal"),
            validator.target_from_label("linux/x86_64/cpu"),
            validator.target_from_label("linux/aarch64/cpu"),
            validator.target_from_label("linux/aarch64/cuda12"),
            validator.target_from_label("linux/aarch64/cuda13"),
            validator.target_from_label("linux/x86_64/cuda12"),
            validator.target_from_label("linux/x86_64/cuda13"),
            validator.target_from_label("linux/x86_64/rocm"),
            validator.target_from_label("linux/x86_64/vulkan"),
            validator.target_from_label("windows/x86_64/cpu"),
            validator.target_from_label("windows/x86_64/cuda12"),
            validator.target_from_label("windows/x86_64/rocm"),
            validator.target_from_label("windows/x86_64/vulkan"),
        }

        violations = validator.find_matrix_violations(
            assets,
            manifest,
            required_targets,
        )

        self.assertEqual(violations, [])

    def test_explicit_release_native_targets_still_require_configured_entries(self):
        validator = load_validator()
        manifest = {
            "artifacts": [
                {
                    "id": "meshllm-native-runtime-linux-x86_64-cpu",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {"kind": "cpu"},
                }
            ]
        }
        required_targets = {
            validator.target_from_label("linux/x86_64/cpu"),
            validator.target_from_label("linux/aarch64/cpu"),
        }

        violations = validator.find_matrix_violations([], manifest, required_targets)

        self.assertEqual(
            violations,
            ["missing native runtime for binary target linux/aarch64/cpu"],
        )

    def test_explicit_empty_required_targets_disable_asset_inference(self):
        validator = load_validator()
        assets = ["mesh-llm-v0.72.0-rc5-aarch64-unknown-linux-gnu.tar.gz"]

        violations = validator.find_matrix_violations(assets, {}, set())

        self.assertEqual(violations, [])

    def test_cli_accepts_explicit_targets_without_asset_arguments(self):
        validator = load_validator()
        manifest = {
            "artifacts": [
                {
                    "id": "meshllm-native-runtime-linux-x86_64-cpu",
                    "platform": {"os": "linux", "arch": "x86_64"},
                    "backend": {"kind": "cpu"},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = pathlib.Path(directory) / "native-runtimes.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validator.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--required-target",
                    "linux/x86_64/cpu",
                ]
            )

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
