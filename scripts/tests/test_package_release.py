import json
import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "package-release.sh"


def run_bash(command: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {command}'],
        cwd=ROOT,
        env={**os.environ, **env},
        check=False,
        text=True,
        capture_output=True,
    )


class PackageReleaseTests(unittest.TestCase):
    def runtime(self, root: pathlib.Path, runtime_id: str, backend: str) -> pathlib.Path:
        runtime = root / runtime_id
        (runtime / "lib").mkdir(parents=True)
        (runtime / "lib" / "libllama.so").write_bytes(b"runtime")
        (runtime / "README.md").write_text("runtime\n", encoding="utf-8")
        (runtime / "manifest.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "id": runtime_id,
                        "mesh_version": "0.73.1",
                        "skippy_abi": "0.1.0",
                        "platform": {
                            "os": "linux",
                            "arch": "x86_64",
                            "target": "x86_64-unknown-linux-gnu",
                        },
                        "backend": {"kind": backend},
                        "rank": 0,
                        "libraries": ["lib/libllama.so"],
                        "url": None,
                        "sha256": None,
                        "signature": None,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return runtime

    def test_selects_exact_platform_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            selected = self.runtime(root, "linux-cpu", "cpu")
            self.runtime(root, "linux-vulkan", "vulkan")
            result = run_bash(
                "select_native_runtime_dir",
                {
                    "MESH_LLM_NATIVE_RUNTIME_ROOT": str(root),
                    "MESH_RELEASE_OS": "Linux",
                    "MESH_RELEASE_ARCH": "x86_64",
                    "MESH_RELEASE_FLAVOR": "cpu",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(pathlib.Path(result.stdout.strip()), selected)

    def test_rejects_ambiguous_runtime_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.runtime(root, "linux-cpu-a", "cpu")
            self.runtime(root, "linux-cpu-b", "cpu")
            result = run_bash(
                "select_native_runtime_dir",
                {
                    "MESH_LLM_NATIVE_RUNTIME_ROOT": str(root),
                    "MESH_RELEASE_OS": "Linux",
                    "MESH_RELEASE_ARCH": "x86_64",
                    "MESH_RELEASE_FLAVOR": "cpu",
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly one native runtime", result.stderr)

    def test_writes_product_v2_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(runtime_root, "linux-cpu", "cpu")
            result = run_bash(
                (
                    f'write_product_manifest "{bundle}" "{host}" "{runtime}" '
                    '"v0.73.1" "cpu"'
                ),
                {},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (bundle / "product-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["contract"], "mesh-llm-product-v2")
            self.assertEqual(manifest["mesh_version"], "0.73.1")
            self.assertEqual(manifest["host"]["path"], "mesh-llm")
            self.assertEqual(
                manifest["runtime"]["path"], "native-runtimes/linux-cpu"
            )


if __name__ == "__main__":
    unittest.main()
