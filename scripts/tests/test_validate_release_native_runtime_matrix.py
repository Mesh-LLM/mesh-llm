import importlib.util
import pathlib
import sys
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
    def test_linux_aarch64_bundles_require_matching_runtime_entries(self):
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


if __name__ == "__main__":
    unittest.main()
