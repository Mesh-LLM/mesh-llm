from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BuildAcceleratorDefaultsTests(unittest.TestCase):
    def test_workflows_do_not_bypass_repository_defaults(self) -> None:
        findings: list[str] = []
        for path in sorted((ROOT / ".github").rglob("*.yml")):
            text = path.read_text(encoding="utf-8")
            for forbidden in (
                'RUSTC_WRAPPER: ""',
                "RUSTC_WRAPPER: ''",
                'LLAMA_STAGE_USE_SCCACHE: "0"',
                "LLAMA_STAGE_USE_SCCACHE: '0'",
                "fuse-ld=lld",
                "-C linker=",
                "-Clinker=",
            ):
                if forbidden in text:
                    findings.append(f"{path.relative_to(ROOT)}: {forbidden}")
            if re.search(r"CARGO_TARGET_[A-Z0-9_]+_LINKER", text):
                findings.append(f"{path.relative_to(ROOT)}: target linker environment override")
        self.assertEqual(findings, [])

    def test_linux_and_windows_release_paths_keep_sccache_enabled(self) -> None:
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        short_paths = (
            ROOT / ".github/actions/setup-windows-short-paths/action.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("use_sccache", release)
        self.assertIn('SCCACHE_DIR = "C:\\s"', short_paths)
        self.assertIn('TEMP = "C:\\t"', short_paths)
        self.assertIn('MESH_LLM_REQUIRE_SCCACHE: "1"', release)
        self.assertIn("New-Item -ItemType Directory -Force", short_paths)

    def test_all_managed_windows_compile_jobs_configure_short_paths(self) -> None:
        workflows = {
            "ci-platform-checks-slice.yml": 1,
            "ci-windows-host-slice.yml": 1,
            "ci-windows-runtime-slice.yml": 1,
            "node-sdk-addon-artifact.yml": 1,
            "release.yml": 3,
            "windows-warm-caches.yml": 2,
        }
        for name, expected in workflows.items():
            with self.subTest(workflow=name):
                source = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertEqual(source.count("uses: ./.github/actions/setup-windows-short-paths"), expected)

    def test_every_windows_native_backend_hashes_long_object_paths(self) -> None:
        bash_build = (ROOT / "scripts/build-llama.sh").read_text(encoding="utf-8")
        powershell_build = (ROOT / "scripts/build-windows.ps1").read_text(encoding="utf-8")
        self.assertIn("-DCMAKE_OBJECT_PATH_MAX=180", bash_build)
        self.assertIn('"-DCMAKE_OBJECT_PATH_MAX=180"', powershell_build)
        rocm_case = powershell_build.rsplit('\n        "rocm" {', maxsplit=1)[1]
        self.assertNotIn('"-DCMAKE_OBJECT_PATH_MAX=180"', rocm_case)

    def test_linker_drivers_encode_platform_policy(self) -> None:
        unix = (ROOT / "scripts/cargo-linker").read_text(encoding="utf-8")
        windows = (ROOT / "scripts/cargo-linker.cmd").read_text(encoding="utf-8")
        self.assertIn("command -v mold", unix)
        self.assertIn("probe_linker_cached mold", unix)
        self.assertIn("report_lld_fallback", unix)
        self.assertIn("xcrun --show-sdk-build-version", unix)
        self.assertIn("rust-lld.exe", windows)
        self.assertIn("lld-link.exe", windows)
        self.assertIn("aarch64-linux-gnu-gcc", unix)
        self.assertIn("x86_64-linux-gnu-gcc", unix)

    def test_developer_bootstrap_pins_sccache_and_installs_linkers(self) -> None:
        unix = (ROOT / "scripts/bootstrap-build-tools").read_text(encoding="utf-8")
        windows = (ROOT / "scripts/bootstrap-build-tools.ps1").read_text(encoding="utf-8")
        recipes = (ROOT / "just/build.just").read_text(encoding="utf-8")
        self.assertIn('MESH_LLM_SCCACHE_VERSION:-0.16.0', unix)
        self.assertIn("mold lld", unix)
        self.assertIn("brew install lld", unix)
        self.assertIn('"0.16.0"', windows)
        self.assertIn("rustup component add llvm-tools-preview", windows)
        self.assertEqual(recipes.count("bootstrap-build-tools:"), 2)


if __name__ == "__main__":
    unittest.main()
