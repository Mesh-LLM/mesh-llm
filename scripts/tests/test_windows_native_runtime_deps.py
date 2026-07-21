import importlib.util
import json
import os
import pathlib
import shutil
import struct
import subprocess
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "windows-native-runtime-deps.py"
VERIFY_SCRIPT = pathlib.Path(__file__).parents[1] / "verify-native-runtime-package.sh"
SPEC = importlib.util.spec_from_file_location("windows_native_runtime_deps", SCRIPT)
DEPS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEPS)


def bash_executable() -> str:
    if os.name != "nt":
        return shutil.which("bash") or "bash"
    git = shutil.which("git")
    if git:
        candidate = pathlib.Path(git).parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Git Bash is required for native-runtime verifier tests")


def write_pe(path: pathlib.Path, imports: list[str]) -> None:
    data = bytearray(0x800)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x8664, 1, 0, 0, 0, 0xF0, 0)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<II", data, optional + 120, 0x1000, (len(imports) + 1) * 20)
    section = optional + 0xF0
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x500, 0x1000, 0x500, 0x200)
    name_offset = 0x300
    for index, name in enumerate(imports):
        descriptor = 0x200 + index * 20
        name_rva = 0x1000 + name_offset - 0x200
        struct.pack_into("<IIIII", data, descriptor, 0, 0, 0, name_rva, 0)
        encoded = name.encode("ascii") + b"\0"
        data[name_offset : name_offset + len(encoded)] = encoded
        name_offset += len(encoded)
    path.write_bytes(data)


class WindowsNativeRuntimeDepsTests(unittest.TestCase):
    def test_reads_pe_import_table(self):
        with tempfile.TemporaryDirectory() as directory:
            library = pathlib.Path(directory) / "ggml-vulkan.dll"
            write_pe(library, ["KERNEL32.dll", "libstdc++-6.dll"])

            self.assertEqual(
                DEPS.imported_dlls(library), ["KERNEL32.dll", "libstdc++-6.dll"]
            )

    def test_collects_mingw_dependency_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lib_dir = root / "artifact" / "lib"
            search_dir = root / "sdk" / "Bin"
            lib_dir.mkdir(parents=True)
            search_dir.mkdir(parents=True)
            write_pe(
                lib_dir / "ggml-vulkan.dll",
                ["vulkan-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll"],
            )
            write_pe(
                search_dir / "libstdc++-6.dll",
                ["KERNEL32.dll", "libgcc_s_seh-1.dll"],
            )
            write_pe(search_dir / "libwinpthread-1.dll", ["KERNEL32.dll"])
            write_pe(search_dir / "libgcc_s_seh-1.dll", ["KERNEL32.dll"])

            copied = DEPS.collect_dependencies(lib_dir, [search_dir])

            self.assertEqual(
                {path.name for path in copied},
                {"libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll"},
            )
            DEPS.verify_dependencies(lib_dir)

    def test_verification_rejects_missing_non_system_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            lib_dir = pathlib.Path(directory)
            write_pe(lib_dir / "ggml-vulkan.dll", ["libstdc++-6.dll"])

            with self.assertRaisesRegex(RuntimeError, "libstdc\\+\\+-6.dll"):
                DEPS.verify_dependencies(lib_dir)

    def test_package_verifier_accepts_closed_windows_dependency_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = pathlib.Path(directory) / "meshllm-native-runtime-windows-x86_64-vulkan"
            lib_dir = artifact / "lib"
            lib_dir.mkdir(parents=True)
            write_pe(lib_dir / "ggml-vulkan.dll", ["vulkan-1.dll", "libstdc++-6.dll"])
            write_pe(lib_dir / "libstdc++-6.dll", ["KERNEL32.dll"])
            write_pe(lib_dir / "llama.dll", ["ggml-vulkan.dll"])
            manifest = {
                "runtime": {
                    "id": artifact.name,
                    "mesh_version": "0.72.1",
                    "skippy_abi": "0.1.32",
                    "platform": {"os": "windows", "arch": "x86_64"},
                    "backend": {"kind": "vulkan"},
                    "libraries": [
                        "lib/ggml-vulkan.dll",
                        "lib/libstdc++-6.dll",
                        "lib/llama.dll",
                    ],
                }
            }
            (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = subprocess.run(
                [bash_executable(), VERIFY_SCRIPT.as_posix(), artifact.as_posix()],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("verified native runtime artifact", result.stdout)


if __name__ == "__main__":
    unittest.main()
