import hashlib
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
    def test_accepts_explicit_immutable_host_input_directory(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('RELEASE_BIN_DIR="${MESH_LLM_RELEASE_BIN_DIR:-$REPO_ROOT/target/release}"', script)
        self.assertIn('MESH_RELEASE_HOST_PRESTAMPED=1 requires', script)
    def runtime(
        self,
        root: pathlib.Path,
        runtime_id: str,
        backend: str,
        build_backend: str | None = None,
    ) -> pathlib.Path:
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
                    },
                    "build": {"backend": build_backend or backend},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return runtime

    def provider_runtime(
        self,
        bundle: pathlib.Path,
        runtime_id: str = "meshllm-apple-runtime-darwin-arm64",
    ) -> pathlib.Path:
        provider = bundle / "provider-runtimes" / "apple" / runtime_id
        (provider / "bin").mkdir(parents=True)
        executable = provider / "bin" / "mesh-apple-runtime"
        executable.write_bytes(b"provider")
        executable.chmod(0o755)
        (provider / "provider-runtime.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "runtime": {
                        "id": runtime_id,
                        "version": "0.1.0",
                        "provider_kind": "apple",
                        "protocol_version": "0.1",
                        "entrypoint": "bin/mesh-apple-runtime",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return provider

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

    def test_product_manifest_attests_embedded_provider_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(runtime_root, "linux-cpu", "cpu")
            provider = self.provider_runtime(bundle)
            result = run_bash(
                (
                    f'PROVIDER_RUNTIME_BUNDLES=("{provider}"); '
                    f'write_product_manifest "{bundle}" "{host}" "{runtime}" '
                    '"v0.73.1" "cpu"'
                ),
                {},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (bundle / "product-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["provider_runtimes"],
                [
                    {
                        "id": "meshllm-apple-runtime-darwin-arm64",
                        "manifest_sha256": hashlib.sha256(
                            (provider / "provider-runtime.json").read_bytes()
                        ).hexdigest(),
                        "path": (
                            "provider-runtimes/apple/"
                            "meshllm-apple-runtime-darwin-arm64"
                        ),
                        "protocol_version": "0.1",
                        "provider_kind": "apple",
                        "sha256": manifest["provider_runtimes"][0]["sha256"],
                        "version": "0.1.0",
                    }
                ],
            )

    def test_product_manifest_rejects_provider_outside_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(runtime_root, "linux-cpu", "cpu")
            provider = self.provider_runtime(bundle)
            wrong_path = bundle / "provider-runtimes" / "wrong"
            provider.rename(wrong_path)
            result = run_bash(
                (
                    f'PROVIDER_RUNTIME_BUNDLES=("{wrong_path}"); '
                    f'write_product_manifest "{bundle}" "{host}" "{runtime}" '
                    '"v0.73.1" "cpu"'
                ),
                {},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("path mismatch", result.stderr)

    def test_rejects_product_runtime_backend_mismatch_before_manifest_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(runtime_root, "linux-vulkan", "vulkan")
            result = run_bash(
                (
                    f'write_product_manifest "{bundle}" "{host}" "{runtime}" '
                    '"v0.73.1" "cpu"'
                ),
                {},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("backend mismatch", result.stderr)
            self.assertFalse((bundle / "product-manifest.json").exists())

    def test_product_manifest_accepts_cuda_blackwell_backend_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(
                runtime_root,
                "linux-cuda13-sm120",
                "cuda",
                build_backend="cuda-blackwell",
            )
            result = run_bash(
                (
                    f'write_product_manifest "{bundle}" "{host}" "{runtime}" '
                    '"v0.73.1" "cuda-blackwell"'
                ),
                {},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (bundle / "product-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["backend"], "cuda-blackwell")
            self.assertEqual(manifest["runtime"]["id"], "linux-cuda13-sm120")
            self.assertEqual(
                manifest["runtime"]["path"],
                "native-runtimes/linux-cuda13-sm120",
            )

    def test_product_manifest_accepts_hip_backend_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(
                runtime_root,
                "linux-rocm",
                "rocm",
                build_backend="hip",
            )
            result = run_bash(
                (
                    f'write_product_manifest "{bundle}" "{host}" "{runtime}" '
                    '"v0.73.1" "hip"'
                ),
                {},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (bundle / "product-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["backend"], "hip")
            self.assertEqual(manifest["runtime"]["id"], "linux-rocm")
            self.assertEqual(manifest["runtime"]["path"], "native-runtimes/linux-rocm")

    def test_rejects_provider_runtime_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(runtime_root, "linux-cpu", "cpu")
            provider = bundle / "provider-runtimes" / "apple" / "meshllm-apple-runtime-darwin-arm64"
            (provider / "bin").mkdir(parents=True)
            executable = provider / "bin" / "mesh-apple-runtime"
            executable.write_bytes(b"provider")
            executable.chmod(0o755)
            (provider / "provider-runtime.json").write_text(
                json.dumps({"schema_version": 1, "runtime": {"id": "meshllm-apple-runtime-darwin-arm64"}}) + "\n",
                encoding="utf-8",
            )
            result = run_bash(
                f'write_product_manifest "{bundle}" "{host}" "{runtime}" "v0.73.1" "cpu"',
                {},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required field", result.stderr)

    def test_rejects_provider_runtime_with_invalid_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory) / "mesh-bundle"
            runtime_root = bundle / "native-runtimes"
            host = bundle / "mesh-llm"
            bundle.mkdir()
            host.write_bytes(b"host")
            runtime = self.runtime(runtime_root, "linux-cpu", "cpu")
            provider = bundle / "provider-runtimes" / "apple" / "meshllm-apple-runtime-darwin-arm64"
            (provider / "bin").mkdir(parents=True)
            (provider / "provider-runtime.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "runtime": {
                        "id": "meshllm-apple-runtime-darwin-arm64",
                        "version": "0.1.0",
                        "provider_kind": "apple",
                        "protocol_version": "0.1",
                        "entrypoint": "bin/nonexistent",
                    },
                }) + "\n",
                encoding="utf-8",
            )
            result = run_bash(
                f'write_product_manifest "{bundle}" "{host}" "{runtime}" "v0.73.1" "cpu"',
                {},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("entrypoint", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
