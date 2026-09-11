from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "linux-native-runtime-deps.py"
SPEC = importlib.util.spec_from_file_location("linux_native_runtime_deps", SCRIPT)
DEPS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEPS)


class LinuxNativeRuntimeDepsPolicyTests(unittest.TestCase):
    def test_reviewed_redistributables_require_matching_cuda_major(self) -> None:
        for name in (
            "libcudart.so.12",
            "libcublas.so.12.9.0.13",
            "libcublasLt.so.12",
            "libnvJitLink.so.12",
        ):
            DEPS._validate_cuda_redistributable(name, 12)

        with self.assertRaisesRegex(RuntimeError, "allowlist"):
            DEPS._validate_cuda_redistributable("libnvrtc.so.12", 12)
        with self.assertRaisesRegex(RuntimeError, "CUDA 13"):
            DEPS._validate_cuda_redistributable("libcudart.so.12", 13)

    def test_provider_selection_ignores_foreign_architecture_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x86_path = root / "x86" / "libcudart.so.12"
            arm_path = root / "arm" / "libcudart.so.12"
            x86_path.parent.mkdir()
            arm_path.parent.mkdir()
            x86_path.write_bytes(b"x86")
            arm_path.write_bytes(b"arm")
            x86 = DEPS.ElfImage(
                x86_path,
                (),
                "libcudart.so.12",
                "ELF64",
                "Advanced Micro Devices X86-64",
            )
            arm = DEPS.ElfImage(
                arm_path,
                (),
                "libcudart.so.12",
                "ELF64",
                "AArch64",
            )

            selected = DEPS._select_provider(
                "libcudart.so.12", [arm, x86], arch="x86_64"
            )

            self.assertEqual(selected.path, x86_path)

    def test_copy_rejects_existing_symlink_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source" / "libcudart.so.12"
            source_path.parent.mkdir()
            source_path.write_bytes(b"cuda runtime")
            lib_dir = root / "lib"
            lib_dir.mkdir()
            (lib_dir / "libcudart.so.12").symlink_to(source_path)
            source = DEPS.ElfImage(source_path, (), "libcudart.so.12", "", "")

            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                DEPS._copy_dependency(
                    source,
                    "libcudart.so.12",
                    lib_dir,
                    arch=None,
                )

    def test_collection_rejects_an_iteration_without_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source" / "libcudart.so.12"
            source_path.parent.mkdir()
            source_path.write_bytes(b"cuda runtime")
            source = DEPS.ElfImage(source_path, (), "libcudart.so.12", "", "")
            destination = root / "lib" / "libcudart.so.12"
            destination.parent.mkdir()
            gaps = {"libllama.so": {"libcudart.so.12"}}

            with (
                mock.patch.object(DEPS, "dependency_gaps", return_value=gaps),
                mock.patch.object(
                    DEPS,
                    "_search_index",
                    return_value=({"libcudart.so.12": [source]}, {}),
                ),
                mock.patch.object(DEPS, "_copy_dependency", return_value=destination),
            ):
                with self.assertRaisesRegex(RuntimeError, "made no progress"):
                    DEPS.collect_dependencies(
                        destination.parent,
                        [source_path.parent],
                        arch=None,
                        cuda_major=12,
                    )


class LinuxNativeRuntimeDepsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = shutil.which("cc") or shutil.which("gcc")
        if cls.compiler is None or shutil.which("readelf") is None:
            raise unittest.SkipTest(
                "a Linux C compiler and readelf are required for ELF dependency fixtures"
            )

    def build_shared(
        self,
        directory: Path,
        name: str,
        soname: str,
        *,
        links: tuple[str, ...] = (),
        link_directory: Path | None = None,
    ) -> Path:
        source = directory / f"{name}.c"
        output = directory / name
        source.write_text("void mesh_fixture_symbol(void) {}\n", encoding="utf-8")
        command = [
            self.compiler,
            "-shared",
            "-fPIC",
            "-nostdlib",
            "-Wl,--no-as-needed",
            f"-Wl,-soname,{soname}",
            "-o",
            str(output),
            str(source),
        ]
        for link in links:
            command.extend(["-L", str(link_directory or directory), f"-l:{link}"])
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def write_cuda_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        lib_dir = root / "artifact" / "lib"
        tools_dir = root / "artifact" / "tools"
        toolkit_dir = root / "toolkit" / "lib64"
        for path in (lib_dir, tools_dir, toolkit_dir):
            path.mkdir(parents=True)

        self.build_shared(toolkit_dir, "libcublasLt.so.12", "libcublasLt.so.12")
        self.build_shared(
            toolkit_dir,
            "libcublas.so.12",
            "libcublas.so.12",
            links=("libcublasLt.so.12",),
        )
        self.build_shared(toolkit_dir, "libcudart.so.12", "libcudart.so.12")
        primary = self.build_shared(
            lib_dir,
            "libllama.so",
            "libllama.so",
            links=("libcudart.so.12", "libcublas.so.12"),
            link_directory=toolkit_dir,
        )
        return lib_dir, tools_dir, toolkit_dir

    def test_collects_cuda_transitive_closure_and_orders_primary_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lib_dir, tools_dir, toolkit_dir = self.write_cuda_fixture(Path(directory))

            copied = DEPS.collect_dependencies(
                lib_dir,
                [toolkit_dir],
                [lib_dir, tools_dir],
                arch="x86_64",
                cuda_major=12,
            )

            self.assertEqual(
                {path.name for path in copied},
                {"libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"},
            )
            for path in copied:
                source = toolkit_dir / path.name
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).digest(),
                    hashlib.sha256(source.read_bytes()).digest(),
                    f"redistributed CUDA object was modified: {path.name}",
                )
            DEPS.verify_dependencies(lib_dir, [lib_dir, tools_dir], arch="x86_64")
            ordered = DEPS.dependency_order(
                lib_dir,
                [lib_dir, tools_dir],
                primary="libllama.so",
                arch="x86_64",
            )
            self.assertEqual(ordered[-1].name, "libllama.so")
            self.assertEqual(
                [path.name for path in ordered].count("libllama.so"),
                1,
            )
            self.assertLess(
                [path.name for path in ordered].index("libcublasLt.so.12"),
                [path.name for path in ordered].index("libcublas.so.12"),
            )

    def test_unresolved_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib_dir = root / "lib"
            lib_dir.mkdir()
            self.build_shared(
                lib_dir,
                "libllama.so",
                "libllama.so",
                links=(),
            )
            # A direct NEEDED entry is enough to exercise the closed-graph
            # verifier without depending on a host CUDA installation.
            result = subprocess.run(
                ["patchelf", "--add-needed", "libcudart.so.12", str(lib_dir / "libllama.so")],
                capture_output=True,
                text=True,
                check=False,
            ) if shutil.which("patchelf") else None
            if result is None:
                self.skipTest("patchelf is required for the unresolved dependency fixture")
            self.assertEqual(result.returncode, 0, result.stderr)
            with self.assertRaisesRegex(RuntimeError, "libcudart\\.so\\.12"):
                DEPS.verify_dependencies(lib_dir, [lib_dir], arch="x86_64")

    def test_stub_is_rejected_only_when_it_is_the_selected_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib_dir, tools_dir, _ = self.write_cuda_fixture(root)
            stub_dir = root / "toolkit" / "targets" / "x86_64" / "lib" / "stubs"
            stub_dir.mkdir(parents=True)
            stub = self.build_shared(stub_dir, "libcudart.so.12", "libcudart.so.12")
            # The real toolkit directory wins while the unrelated stub is
            # ignored. Remove the real provider to prove a needed stub fails.
            (root / "toolkit" / "lib64" / "libcudart.so.12").unlink()
            with self.assertRaisesRegex(RuntimeError, "stub"):
                DEPS.collect_dependencies(
                    lib_dir,
                    [root / "toolkit" / "lib64", stub_dir],
                    [lib_dir, tools_dir],
                    arch="x86_64",
                    cuda_major=12,
                )

    def test_wrong_architecture_provider_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lib_dir = root / "lib"
            provider_dir = root / "provider"
            lib_dir.mkdir()
            provider_dir.mkdir()
            self.build_shared(lib_dir, "libllama.so", "libllama.so")
            source = provider_dir / "libcudart.so.12"
            shutil.copy2("/bin/true", source)
            if not shutil.which("patchelf"):
                self.skipTest("patchelf is required for the wrong architecture fixture")
            result = subprocess.run(
                ["patchelf", "--add-needed", "libcudart.so.12", str(lib_dir / "libllama.so")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with self.assertRaisesRegex(RuntimeError, "wrong architecture"):
                DEPS.collect_dependencies(
                    lib_dir,
                    [provider_dir],
                    [lib_dir],
                    arch="aarch64",
                    cuda_major=12,
                )


if __name__ == "__main__":
    unittest.main()
