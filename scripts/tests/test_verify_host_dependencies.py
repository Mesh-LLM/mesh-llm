from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify-host-dependencies.py"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
SPEC = importlib.util.spec_from_file_location("verify_host_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyHostDependenciesTests(unittest.TestCase):
    def test_shared_host_actions_invoke_non_executable_verifier_with_python(
        self,
    ) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        unix_action = (
            ROOT / ".github" / "actions" / "prepare-host-input" / "action.yml"
        ).read_text(encoding="utf-8")
        windows_action = (
            ROOT
            / ".github"
            / "actions"
            / "prepare-windows-host-input"
            / "action.yml"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            unix_action.count("python3 scripts/verify-host-dependencies.py"),
            1,
        )
        self.assertEqual(
            windows_action.count(
                r"& python scripts\verify-host-dependencies.py",
            ),
            1,
        )
        self.assertNotIn("verify-host-dependencies.py", workflow)

    def test_declared_linux_glibc_floor_is_a_single_parseable_version(self) -> None:
        floor = MODULE.read_declared_glibc_floor(
            ROOT / "scripts" / "linux-glibc-floor.txt"
        )

        self.assertEqual(len(floor), 2)
        self.assertGreaterEqual(floor, (2, 17))

    def test_glibc_floor_reads_only_the_version_needs_section(self) -> None:
        # A shared library defines its own versions in the same readelf
        # output. Counting those would report a floor the loader never checks.
        output = """
Version definition section '.gnu.version_d' contains 2 entries:
  000000: Rev: 1  Flags: BASE  Index: 1  Cnt: 1  Name: GLIBC_2.99

Version needs section '.gnu.version_r' contains 1 entry:
  0x0000: Version: 1  File: libc.so.6  Cnt: 3
  0x0010:   Name: GLIBC_2.14  Flags: none  Version: 4
  0x0020:   Name: GLIBC_2.38  Flags: none  Version: 3
  0x0030:   Name: GLIBC_2.4  Flags: none  Version: 2
"""

        self.assertEqual(MODULE.parse_elf_glibc_floor(output), (2, 38))

    def test_glibc_floor_orders_versions_numerically_not_lexically(self) -> None:
        output = """
Version needs section '.gnu.version_r' contains 1 entry:
  0x0010:   Name: GLIBC_2.9  Flags: none  Version: 3
  0x0020:   Name: GLIBC_2.34  Flags: none  Version: 2
"""

        self.assertEqual(MODULE.parse_elf_glibc_floor(output), (2, 34))

    def test_binaries_without_a_version_needs_section_have_no_floor(self) -> None:
        self.assertIsNone(MODULE.parse_elf_glibc_floor("no versions here"))

    def test_max_glibc_accepts_a_literal_version_or_the_declared_file(self) -> None:
        self.assertEqual(MODULE.resolve_max_glibc("2.35"), (2, 35))
        self.assertIsNone(MODULE.resolve_max_glibc(None))
        self.assertEqual(
            MODULE.resolve_max_glibc("declared"),
            MODULE.read_declared_glibc_floor(
                ROOT / "scripts" / "linux-glibc-floor.txt"
            ),
        )

    def test_host_and_runtime_lanes_both_enforce_the_declared_floor(self) -> None:
        unix_action = (
            ROOT / ".github" / "actions" / "prepare-host-input" / "action.yml"
        ).read_text(encoding="utf-8")
        runtime_verifier = (
            ROOT / "scripts" / "verify-native-runtime-package.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--max-glibc declared", unix_action)
        self.assertIn("--max-glibc declared", runtime_verifier)
        self.assertIn("verify_linux_glibc_floor", runtime_verifier)
        # Tools ship in the package too and the host executes the GPU
        # benchmark, so enumerating only libraries let an over-floor tool
        # through. Behavior is covered in test_native_artifact_verifiers.
        self.assertIn(
            'for rel_path in [*runtime["libraries"], *(runtime.get("tools") or {})]:',
            runtime_verifier,
        )
        # Runtime libraries import each other, so the host policy is off for
        # them and stays on for the host.
        self.assertIn("--no-import-policy", runtime_verifier)
        self.assertNotIn("--no-import-policy", unix_action)

    def test_parses_elf_needed_entries(self) -> None:
        imports = MODULE.parse_elf_imports(
            """
 0x0000000000000001 (NEEDED)             Shared library: [libSystem.so]
 0x0000000000000001 (NEEDED)             Shared library: [libcuda.so.1]
"""
        )

        self.assertEqual(imports, ["libSystem.so", "libcuda.so.1"])

    def test_parses_macho_and_pe_imports(self) -> None:
        macho = MODULE.parse_macho_imports(
            """
mesh-llm:
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)
\t/System/Library/Frameworks/Metal.framework/Versions/A/Metal (compatibility version 1.0.0, current version 1.0.0)
"""
        )
        pe = MODULE.parse_pe_imports(
            """
    Name: KERNEL32.dll
        DLL Name: vulkan-1.dll
"""
        )

        self.assertEqual(
            macho,
            [
                "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
                "/usr/lib/libSystem.B.dylib",
            ],
        )
        self.assertEqual(pe, ["KERNEL32.dll", "vulkan-1.dll"])

    def test_rejects_backend_imports_but_allows_host_system_libraries(self) -> None:
        imports = [
            "libc.so.6",
            "libcuda.so.1",
            "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
            "vulkan-1.dll",
            "libllama.so",
        ]

        self.assertEqual(
            MODULE.forbidden_imports(imports),
            imports[1:],
        )

    def test_rejects_windows_cuda_and_rocm_runtime_dll_names(self) -> None:
        imports = [
            "KERNEL32.dll",
            "nvcuda.dll",
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "amdhip64.dll",
            "hipblas.dll",
            "rocblas.dll",
        ]

        self.assertEqual(MODULE.forbidden_imports(imports), imports[1:])


if __name__ == "__main__":
    unittest.main()
