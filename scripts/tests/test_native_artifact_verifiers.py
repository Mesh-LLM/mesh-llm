from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_VERIFIER = ROOT / "scripts" / "verify-native-runtime-package.sh"
SDK_VERIFIER = ROOT / "scripts" / "verify-native-sdk-package.sh"


def bash_executable() -> str:
    if os.name != "nt":
        return shutil.which("bash") or "bash"
    git = shutil.which("git")
    if git:
        candidate = Path(git).parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Git Bash is required for artifact verifier tests")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeArtifactVerifierTests(unittest.TestCase):
    def run_verifier(
        self,
        verifier: Path,
        artifact: Path,
        *options: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                bash_executable(),
                verifier.as_posix(),
                *options,
                artifact.as_posix(),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def write_runtime_artifact(self, root: Path) -> tuple[Path, dict]:
        artifact = root / "meshllm-native-runtime-darwin-x86_64-cpu"
        library = artifact / "lib" / "llama.bin"
        tool = artifact / "tools" / "probe"
        library.parent.mkdir(parents=True)
        tool.parent.mkdir(parents=True)
        library.write_bytes(b"runtime library")
        tool.write_bytes(b"runtime tool")
        tool.chmod(0o755)
        manifest = {
            "runtime": {
                "id": artifact.name,
                "mesh_version": "0.75.0",
                "skippy_abi": "0.1.32",
                "platform": {
                    "os": "macos",
                    "arch": "x86_64",
                    "target": "x86_64-apple-darwin",
                },
                "backend": {"kind": "cpu"},
                "libraries": ["lib/llama.bin"],
                "files": {"lib/llama.bin": sha256(library)},
                "tools": {"tools/probe": sha256(tool)},
            },
            "build": {
                "primary_library": "lib/llama.bin",
                "library_sha256": sha256(library),
            },
        }
        self.write_manifest(artifact, manifest)
        return artifact, manifest

    def write_sdk_artifact(self, root: Path) -> tuple[Path, dict]:
        artifact = root / "meshllm-native-linux-x86_64-cpu"
        library = artifact / "lib" / "libmesh_llm_ffi.so"
        uniffi_library = artifact / "lib" / "libmesh_llm_uniffi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"native SDK library")
        uniffi_library.write_bytes(library.read_bytes())
        manifest = {
            "schema_version": 1,
            "artifact_id": artifact.name,
            "native_runtime_id": artifact.name,
            "sdk_version": "0.75.0",
            "mesh_version": "0.75.0",
            "target_triple": "x86_64-unknown-linux-gnu",
            "platform": "linux-x86_64",
            "os": "linux",
            "arch": "x86_64",
            "backend": "cpu",
            "flavor": "cpu",
            "library": "lib/libmesh_llm_ffi.so",
            "library_paths": ["lib/libmesh_llm_ffi.so"],
            "uniffi_library": "lib/libmesh_llm_uniffi.so",
            "library_sha256": sha256(library),
            "requirements": [],
            "features": [
                "mesh-inference",
                "model-management",
                "local-serving",
                "chat",
                "responses",
            ],
        }
        self.write_manifest(artifact, manifest)
        return artifact, manifest

    def write_manifest(self, artifact: Path, manifest: dict) -> None:
        (artifact / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def archive_artifact(
        self,
        artifact: Path,
        archive: Path,
        *,
        sibling_payload: bool = False,
    ) -> None:
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(artifact, arcname=artifact.name)
            if sibling_payload:
                payload = b"unexpected sibling"
                sibling = tarfile.TarInfo("unexpected.txt")
                sibling.size = len(payload)
                bundle.addfile(sibling, io.BytesIO(payload))
        archive.with_name(f"{archive.name}.sha256").write_text(
            f"{sha256(archive)}  {archive.name}\n",
            encoding="utf-8",
        )

    def test_runtime_rejects_every_unsafe_manifest_path(self) -> None:
        def replace_library(manifest: dict, path: str) -> None:
            runtime = manifest["runtime"]
            runtime["libraries"] = [path]
            runtime["files"] = {path: "0" * 64}
            manifest["build"]["primary_library"] = path
            manifest["build"]["library_sha256"] = "0" * 64

        def replace_primary(manifest: dict, path: str) -> None:
            runtime = manifest["runtime"]
            runtime["libraries"].append(path)
            runtime["files"][path] = "0" * 64
            manifest["build"]["primary_library"] = path
            manifest["build"]["library_sha256"] = "0" * 64

        cases = {
            "library traversal": lambda manifest: replace_library(
                manifest,
                "../outside.bin",
            ),
            "file traversal": lambda manifest: manifest["runtime"][
                "files"
            ].update(
                {"../outside.bin": "0" * 64}
            ),
            "tool traversal": lambda manifest: manifest["runtime"].update(
                {"tools": {"../outside.bin": "0" * 64}}
            ),
            "primary traversal": lambda manifest: replace_primary(
                manifest,
                "../outside.bin",
            ),
            "backslash": lambda manifest: replace_library(
                manifest,
                r"lib\llama.bin",
            ),
            "drive": lambda manifest: replace_primary(
                manifest,
                "C:/outside.bin",
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact, manifest = self.write_runtime_artifact(root)
                (root / "outside.bin").write_bytes(b"outside")
                mutate(manifest)
                self.write_manifest(artifact, manifest)

                result = self.run_verifier(RUNTIME_VERIFIER, artifact)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("artifact", result.stdout + result.stderr)

    def test_sdk_rejects_every_unsafe_manifest_path(self) -> None:
        def mutate_primary(manifest: dict) -> None:
            manifest["library"] = "../outside.so"
            manifest["library_paths"] = ["../outside.so"]

        cases = {
            "library traversal": mutate_primary,
            "library_paths traversal": lambda manifest: manifest[
                "library_paths"
            ].append("../outside.so"),
            "uniffi traversal": lambda manifest: manifest.update(
                {"uniffi_library": "../outside.so"}
            ),
            "backslash": lambda manifest: manifest["library_paths"].append(
                r"lib\outside.so"
            ),
            "drive": lambda manifest: manifest.update(
                {"uniffi_library": "C:/outside.so"}
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact, manifest = self.write_sdk_artifact(root)
                (root / "outside.so").write_bytes(b"outside")
                mutate(manifest)
                self.write_manifest(artifact, manifest)

                result = self.run_verifier(SDK_VERIFIER, artifact)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("artifact", result.stdout + result.stderr)

    def test_runtime_rejects_non_string_platform_discriminator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact, manifest = self.write_runtime_artifact(Path(directory))
            manifest["runtime"]["platform"]["os"] = ["macos"]
            self.write_manifest(artifact, manifest)

            result = self.run_verifier(RUNTIME_VERIFIER, artifact)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "runtime platform must declare os and arch",
                result.stderr,
            )

    def test_runtime_requires_complete_library_checksum_contract(self) -> None:
        cases = {
            "missing files": (
                lambda manifest: manifest["runtime"].pop("files"),
                "missing runtime manifest field",
            ),
            "missing library checksum": (
                lambda manifest: manifest["runtime"].update(
                    {"files": {}}
                ),
                "checksum",
            ),
            "missing primary": (
                lambda manifest: manifest["build"].pop(
                    "primary_library"
                ),
                "build.primary_library",
            ),
            "missing primary checksum": (
                lambda manifest: manifest["build"].pop(
                    "library_sha256"
                ),
                "build.library_sha256",
            ),
        }
        for name, (mutate, expected_error) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                artifact, manifest = self.write_runtime_artifact(
                    Path(directory),
                )
                mutate(manifest)
                self.write_manifest(artifact, manifest)

                result = self.run_verifier(
                    RUNTIME_VERIFIER,
                    artifact,
                    "--portable",
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_runtime_rejects_target_and_backend_platform_mismatch(
        self,
    ) -> None:
        cases = {
            "target": lambda manifest: manifest["runtime"][
                "platform"
            ].update({"target": "aarch64-unknown-linux-gnu"}),
            "backend": lambda manifest: (
                manifest["runtime"].update(
                    {"backend": {"kind": "metal"}},
                ),
                manifest["runtime"].update(
                    {
                        "platform": {
                            "os": "linux",
                            "arch": "x86_64",
                            "target": "x86_64-unknown-linux-gnu",
                        }
                    },
                ),
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                artifact, manifest = self.write_runtime_artifact(
                    Path(directory),
                )
                mutate(manifest)
                self.write_manifest(artifact, manifest)

                result = self.run_verifier(
                    RUNTIME_VERIFIER,
                    artifact,
                    "--portable",
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "do not match target"
                    if name == "target"
                    else "unsupported on",
                    result.stderr,
                )

    def test_portable_runtime_verification_skips_host_binary_probes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, manifest = self.write_runtime_artifact(root)
            old_library = artifact / "lib" / "llama.bin"
            library = artifact / "lib" / "libllama.so"
            old_library.rename(library)
            library_digest = sha256(library)
            runtime = manifest["runtime"]
            runtime["platform"] = {
                "os": "linux",
                "arch": "x86_64",
                "target": "x86_64-unknown-linux-gnu",
            }
            runtime["libraries"] = ["lib/libllama.so"]
            runtime["files"] = {"lib/libllama.so": library_digest}
            manifest["build"]["primary_library"] = "lib/libllama.so"
            manifest["build"]["library_sha256"] = library_digest
            self.write_manifest(artifact, manifest)

            result = self.run_verifier(
                RUNTIME_VERIFIER,
                artifact,
                "--portable",
            )

            self.assertEqual(
                result.returncode,
                0,
                result.stdout + result.stderr,
            )
            self.assertIn(
                "verified portable native runtime artifact",
                result.stdout,
            )

    def test_sdk_rejects_unknown_target_backend_and_flavor(self) -> None:
        cases = {
            "target": lambda manifest: manifest.update(
                {"target_triple": "mystery-vendor-platform"}
            ),
            "backend": lambda manifest: manifest.update(
                {"backend": "made-up"}
            ),
            "flavor": lambda manifest: manifest.update(
                {"flavor": "made-up"}
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                artifact, manifest = self.write_sdk_artifact(
                    Path(directory),
                )
                mutate(manifest)
                if name == "flavor":
                    renamed = artifact.with_name(
                        "meshllm-native-linux-x86_64-made-up",
                    )
                    artifact.rename(renamed)
                    artifact = renamed
                    manifest["artifact_id"] = artifact.name
                    manifest["native_runtime_id"] = artifact.name
                self.write_manifest(artifact, manifest)

                result = self.run_verifier(SDK_VERIFIER, artifact)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "unsupported" if name != "flavor" else "flavor",
                    result.stderr,
                )

    def test_sdk_manifest_omits_runner_local_build_directory(self) -> None:
        packager = (
            ROOT / "scripts" / "package-native-sdk.sh"
        ).read_text(encoding="utf-8")

        self.assertNotIn('"llama_build_dir"', packager)
        self.assertNotIn(
            'os.path.abspath("$LLAMA_STAGE_BUILD_DIR")',
            packager,
        )

    @unittest.skipIf(os.name == "nt", "symlink creation is restricted on Windows")
    def test_resolved_manifest_paths_cannot_escape_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            runtime, runtime_manifest = self.write_runtime_artifact(root)
            runtime_link = runtime / "lib" / "escape.bin"
            runtime_link.symlink_to(outside)
            runtime_manifest["runtime"]["libraries"] = ["lib/escape.bin"]
            self.write_manifest(runtime, runtime_manifest)

            runtime_result = self.run_verifier(RUNTIME_VERIFIER, runtime)

            self.assertNotEqual(runtime_result.returncode, 0)
            self.assertIn("resolves outside", runtime_result.stderr)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.so"
            outside.write_bytes(b"outside")
            sdk, sdk_manifest = self.write_sdk_artifact(root)
            sdk_link = sdk / "lib" / "escape.so"
            sdk_link.symlink_to(outside)
            sdk_manifest["library_paths"].append("lib/escape.so")
            self.write_manifest(sdk, sdk_manifest)

            sdk_result = self.run_verifier(SDK_VERIFIER, sdk)

            self.assertNotEqual(sdk_result.returncode, 0)
            self.assertIn("resolves outside", sdk_result.stderr)

    def test_archive_wrappers_accept_one_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (
                    RUNTIME_VERIFIER,
                    self.write_runtime_artifact(root)[0],
                    root / "runtime.tar.gz",
                ),
                (
                    SDK_VERIFIER,
                    self.write_sdk_artifact(root)[0],
                    root / "sdk.tar.gz",
                ),
            )
            for verifier, artifact, archive in cases:
                with self.subTest(verifier=verifier.name):
                    self.archive_artifact(artifact, archive)
                    result = self.run_verifier(verifier, archive)
                    self.assertEqual(
                        result.returncode,
                        0,
                        result.stdout + result.stderr,
                    )

    def test_archive_wrappers_reject_sibling_top_level_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (
                    RUNTIME_VERIFIER,
                    self.write_runtime_artifact(root)[0],
                    root / "runtime.tar.gz",
                ),
                (
                    SDK_VERIFIER,
                    self.write_sdk_artifact(root)[0],
                    root / "sdk.tar.gz",
                ),
            )
            for verifier, artifact, archive in cases:
                with self.subTest(verifier=verifier.name):
                    self.archive_artifact(
                        artifact,
                        archive,
                        sibling_payload=True,
                    )
                    result = self.run_verifier(verifier, archive)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "one top-level artifact directory",
                        result.stdout + result.stderr,
                    )

    def test_archive_wrappers_reject_single_top_level_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for verifier in (RUNTIME_VERIFIER, SDK_VERIFIER):
                with self.subTest(verifier=verifier.name):
                    archive = root / f"{verifier.stem}.tar.gz"
                    with tarfile.open(archive, "w:gz") as bundle:
                        payload = b"not an artifact directory"
                        member = tarfile.TarInfo("payload")
                        member.size = len(payload)
                        bundle.addfile(member, io.BytesIO(payload))
                    archive.with_name(f"{archive.name}.sha256").write_text(
                        f"{sha256(archive)}  {archive.name}\n",
                        encoding="utf-8",
                    )

                    result = self.run_verifier(verifier, archive)

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "one top-level artifact directory",
                        result.stdout + result.stderr,
                    )


if __name__ == "__main__":
    unittest.main()
