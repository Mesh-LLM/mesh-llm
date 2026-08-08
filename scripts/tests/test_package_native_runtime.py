from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "package-native-runtime.sh"


class PackageNativeRuntimeTests(unittest.TestCase):
    def test_cpu_package_with_no_tools_is_safe_under_macos_bash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "libllama.so").write_bytes(b"test native runtime")

            tool_dir = root / "tools"
            tool_dir.mkdir()
            patchelf = tool_dir / "patchelf"
            patchelf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            patchelf.chmod(0o755)

            env = os.environ.copy()
            env["LLAMA_STAGE_BUILD_DIR"] = str(build_dir)
            env["PATH"] = f"{tool_dir}{os.pathsep}{env['PATH']}"
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(SCRIPT),
                    "--backend",
                    "cpu",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "--out",
                    str(root / "output"),
                ],
                env=env,
                text=True,
                capture_output=True,
            )

            result.check_returncode()
            manifest = json.loads(
                (
                    root
                    / "output"
                    / "meshllm-native-runtime-linux-x86_64-cpu"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["runtime"]["tools"], {})

    def test_rocm_benchmark_tool_uses_configured_offload_arches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "libllama.so").write_bytes(b"test native runtime")

            tool_dir = root / "bin"
            tool_dir.mkdir()
            hipcc = tool_dir / "hipcc"
            hipcc.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "printf '%s\\n' \"$@\" > \"$HIPCC_ARGS_LOG\"\n"
                "previous=''\n"
                "for argument in \"$@\"; do\n"
                "  if [ \"$previous\" = '-o' ]; then output=\"$argument\"; fi\n"
                "  previous=\"$argument\"\n"
                "done\n"
                ": > \"$output\"\n",
                encoding="utf-8",
            )
            hipcc.chmod(0o755)
            patchelf = tool_dir / "patchelf"
            patchelf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            patchelf.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HIPCC": str(hipcc),
                    "HIPCC_ARGS_LOG": str(root / "hipcc-args.log"),
                    "LLAMA_STAGE_AMDGPU_TARGETS": "gfx90a;gfx942, gfx1151",
                    "LLAMA_STAGE_BUILD_DIR": str(build_dir),
                    "PATH": f"{tool_dir}{os.pathsep}{env['PATH']}",
                }
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(SCRIPT),
                    "--backend",
                    "rocm",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "--out",
                    str(root / "output"),
                ],
                env=env,
                text=True,
                capture_output=True,
            )

            result.check_returncode()
            arguments = (
                (root / "hipcc-args.log").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(
                [
                    argument
                    for argument in arguments
                    if argument.startswith("--offload-arch=")
                ],
                [
                    "--offload-arch=gfx90a",
                    "--offload-arch=gfx942",
                    "--offload-arch=gfx1151",
                ],
            )
            manifest = json.loads(
                (
                    root
                    / "output"
                    / "meshllm-native-runtime-linux-x86_64-rocm"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["runtime"]["backend"]["rocm"]["gpu_arches"],
                ["gfx90a", "gfx942", "gfx1151"],
            )

    def test_cuda_flavor_uses_mesh_cuda_version_major(self) -> None:
        self.assertEqual(
            self.backend_flavor("cuda", mesh_cuda_version="13.1.2"),
            "cuda13",
        )

    def test_explicit_cuda_toolkit_major_wins(self) -> None:
        self.assertEqual(
            self.backend_flavor(
                "cuda",
                mesh_cuda_version="13.1.2",
                toolkit_major="12",
            ),
            "cuda12",
        )

    def test_cuda_flavor_defaults_to_cuda_12(self) -> None:
        self.assertEqual(self.backend_flavor("cuda"), "cuda12")

    def test_cuda_blackwell_flavor_defaults_to_cuda13_sm120(self) -> None:
        self.assertEqual(self.backend_flavor("cuda-blackwell"), "cuda13-sm120")

    def test_explicit_cuda_toolkit_major_rejects_non_digits(self) -> None:
        for toolkit_major in ("12.1", "cuda12"):
            with self.subTest(toolkit_major=toolkit_major):
                result = self.backend_flavor_process(
                    "cuda",
                    toolkit_major=toolkit_major,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "MESH_LLM_CUDA_TOOLKIT_MAJOR must be digits-only",
                    result.stderr,
                )

    def backend_flavor(
        self,
        backend: str,
        *,
        mesh_cuda_version: str | None = None,
        toolkit_major: str | None = None,
    ) -> str:
        result = self.backend_flavor_process(
            backend,
            mesh_cuda_version=mesh_cuda_version,
            toolkit_major=toolkit_major,
        )
        result.check_returncode()
        return result.stdout.strip()

    def backend_flavor_process(
        self,
        backend: str,
        *,
        mesh_cuda_version: str | None = None,
        toolkit_major: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        script = SCRIPT.read_text(encoding="utf-8")
        start = script.index("backend_flavor()")
        end = script.index("build_backend()", start)
        helpers = script[start:end]
        env = os.environ.copy()
        env["BACKEND"] = backend
        for name, value in (
            ("MESH_CUDA_VERSION", mesh_cuda_version),
            ("MESH_LLM_CUDA_TOOLKIT_MAJOR", toolkit_major),
        ):
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        result = subprocess.run(
            ["bash", "-c", f"set -euo pipefail\n{helpers}\nbackend_flavor"],
            env=env,
            text=True,
            capture_output=True,
        )
        return result


if __name__ == "__main__":
    unittest.main()
