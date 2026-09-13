from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "crates/skippy-ffi/build.rs"
CORE_ARCHIVES = (
    "src/libllama.a",
    "common/libllama-common.a",
    "common/libllama-common-base.a",
    "ggml/src/libggml.a",
    "ggml/src/libggml-base.a",
    "ggml/src/libggml-cpu.a",
)
OPTIONAL_ARCHIVES = (
    "ggml/src/ggml-blas/libggml-blas.a",
    "ggml/src/ggml-cuda/libggml-cuda.a",
    "ggml/src/ggml-hip/libggml-hip.a",
    "ggml/src/ggml-vulkan/libggml-vulkan.a",
    "ggml/src/ggml-metal/libggml-metal.a",
)


class SkippyStaticLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.binary_dir.cleanup)
        cls.binary = Path(cls.binary_dir.name) / "build-script"
        result = subprocess.run(
            ["rustc", "--edition=2024", str(BUILD_SCRIPT), "-o", str(cls.binary)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"build script fixture failed: {result.stderr}")

    def _run(self, backend: str, flags: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return self._run_with_newline(backend, flags, "\n")

    def _run_with_newline(
        self, backend: str, flags: dict[str, str], newline: str
    ) -> subprocess.CompletedProcess[str]:
        fixture = tempfile.TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        build_dir = Path(fixture.name) / "native"
        for relative in (*CORE_ARCHIVES, *OPTIONAL_ARCHIVES):
            archive = build_dir / relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.touch()
        (build_dir / "CMakeCache.txt").write_text(
            "".join(f"{key}:BOOL={value}{newline}" for key, value in flags.items()),
            encoding="utf-8",
            newline="",
        )
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("LLAMA_STAGE_", "SKIPPY_LLAMA_", "CARGO_FEATURE_"))
        }
        env.update({
            "CARGO_MANIFEST_DIR": str(ROOT / "crates/skippy-ffi"),
            "TARGET": "aarch64-apple-darwin",
            "LLAMA_STAGE_BACKEND": backend,
            "LLAMA_STAGE_BUILD_DIR": str(build_dir),
            "LLAMA_STAGE_LINK_MODE": "static",
            "SKIPPY_LLAMA_AUTO_BUILD": "0",
        })
        return subprocess.run(
            [str(self.binary)], cwd=ROOT, env=env,
            capture_output=True, text=True, check=False,
        )

    def test_cpu_ignores_stale_gpu_and_blas_archives(self) -> None:
        result = self._run("cpu", {
            "GGML_BLAS": "OFF", "GGML_CUDA": "OFF", "GGML_HIP": "OFF",
            "GGML_VULKAN": "OFF", "GGML_METAL": "OFF",
        })
        self.assertEqual(0, result.returncode, result.stderr)
        for library in ("ggml-blas", "ggml-cuda", "ggml-hip", "ggml-vulkan", "ggml-metal"):
            self.assertNotIn(f"cargo:rustc-link-lib=static={library}", result.stdout)
        for framework in ("Foundation", "Metal", "MetalKit"):
            self.assertNotIn(f"cargo:rustc-link-lib=framework={framework}", result.stdout)

    def test_active_metal_backend_links_only_cache_enabled_archive(self) -> None:
        result = self._run("metal", {"GGML_METAL": "ON", "GGML_BLAS": "ON"})
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("cargo:rustc-link-lib=static=ggml-metal", result.stdout)
        self.assertIn("cargo:rustc-link-lib=framework=Metal", result.stdout)
        self.assertIn("cargo:rustc-link-lib=static=ggml-blas", result.stdout)
        self.assertNotIn("cargo:rustc-link-lib=static=ggml-cuda", result.stdout)

    def test_backend_cache_mismatch_fails_closed_despite_stale_archive(self) -> None:
        result = self._run("metal", {"GGML_METAL": "OFF"})
        self.assertNotEqual(0, result.returncode)
        self.assertIn("selected backend requires GGML_METAL=ON", result.stderr)

    def test_unselected_backend_cache_mismatch_fails_closed(self) -> None:
        result = self._run("cpu", {"GGML_CUDA": "ON"})
        self.assertNotEqual(0, result.returncode)
        self.assertIn("unselected backend requires GGML_CUDA=OFF", result.stderr)

    def test_crlf_cache_values_are_recognized(self) -> None:
        result = self._run_with_newline("metal", {"GGML_METAL": "ON"}, "\r\n")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("cargo:rustc-link-lib=static=ggml-metal", result.stdout)


if __name__ == "__main__":
    unittest.main()
