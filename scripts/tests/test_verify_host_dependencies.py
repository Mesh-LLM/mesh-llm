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
    def test_release_workflow_uses_python_for_non_executable_verifier(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count("python3 scripts/verify-host-dependencies.py"),
            2,
        )
        self.assertNotIn(
            "\n          scripts/verify-host-dependencies.py",
            workflow,
        )

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
