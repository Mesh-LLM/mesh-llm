from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RESTORE = ROOT / "scripts" / "restore-static-abi-input.sh"
STAMP_VERIFIER = ROOT / "scripts" / "verify-static-abi-build-stamp.py"
TOOLCHAIN_EPOCH = "test-runner-image-sha256-deadbeef"


def bash_executable() -> str:
    if os.name != "nt":
        return shutil.which("bash") or "bash"
    git = shutil.which("git")
    if git:
        candidate = Path(git).parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Git Bash is required for static ABI tests")


def native_linux_target() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64-unknown-linux-gnu"
    if machine in {"amd64", "x86_64"}:
        return "x86_64-unknown-linux-gnu"
    raise unittest.SkipTest(f"unsupported test architecture: {machine}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StaticAbiArtifactTests(unittest.TestCase):
    def test_native_sdk_reuse_is_verification_only(self) -> None:
        build_script = (ROOT / "scripts" / "build-llama.sh").read_text(
            encoding="utf-8",
        )
        package_script = (
            ROOT / "scripts" / "package-native-sdk.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--require-existing", build_script)
        self.assertIn("refusing to rebuild", build_script)
        self.assertIn("toolchain-epoch=", build_script)
        self.assertIn("stamp-version=3", build_script)
        self.assertIn("@LLAMA_BUILD_DIR@", build_script)
        self.assertIn("@LLAMA_WORKDIR@", build_script)
        self.assertIn("-ffile-prefix-map=", build_script)
        self.assertIn("MESH_LLM_REQUIRE_SCCACHE=1", build_script)
        self.assertIn('"$SCCACHE_BIN" --show-stats', build_script)
        self.assertLess(
            build_script.index('"$SCCACHE_BIN" --show-stats'),
            build_script.index('"$SCCACHE_BIN" --start-server'),
        )
        for archive in (
            "libllama-common-base.a",
            "libggml.a",
            "libggml-base.a",
            "libggml-cpu.a",
            "libvendor-hash.a",
        ):
            self.assertIn(archive, build_script)
        self.assertIn("--require-prebuilt-llama", package_script)
        self.assertNotIn("build_args=()", package_script)
        build_section = package_script.split(
            'if [[ "$BUILD" == "1" ]]; then',
            maxsplit=1,
        )[1].split("    cargo_args=", maxsplit=1)[0]
        prebuilt_branch, normal_branch = build_section.split(
            "    else\n",
            maxsplit=1,
        )
        self.assertIn(
            'if [[ "$REQUIRE_PREBUILT_LLAMA" == "1" ]]; then',
            prebuilt_branch,
        )
        self.assertIn(
            '"$SCRIPT_DIR/build-llama.sh" --require-existing',
            prebuilt_branch,
        )
        self.assertIn('"$SCRIPT_DIR/build-llama.sh"', normal_branch)
        self.assertNotIn("--require-existing", normal_branch)
        self.assertIn("SKIPPY_LLAMA_AUTO_BUILD=0", package_script)
        self.assertIn("MESH_LLM_AUTO_BUILD_LLAMA=0", package_script)

    def test_dynamic_output_probe_is_pipefail_safe(self) -> None:
        build_script = (ROOT / "scripts" / "build-llama.sh").read_text(
            encoding="utf-8",
        )
        function = build_script.split(
            "required_dynamic_libraries_exist() {",
            maxsplit=1,
        )[1].split("\n}", maxsplit=1)[0]

        self.assertIn('found="$(find ', function)
        self.assertIn('[[ -n "$found" && -e "$found" ]] || return 1', function)
        self.assertIn("libllama.dll|llama.dll", build_script)
        self.assertIn("libllama-common.dll|llama-common.dll", build_script)
        self.assertIn("libmtmd.dll|mtmd.dll", build_script)
        self.assertNotIn("| grep -q", function)

    def write_artifact(
        self,
        root: Path,
        *,
        manifest_target: str | None = None,
        link_mode: str = "static",
        sibling: bool = False,
        toolchain_epoch: str = TOOLCHAIN_EPOCH,
        omit_archive: str | None = None,
    ) -> Path:
        target = native_linux_target()
        build_dir = root / "source" / "build-stage-abi-static"
        for relative in (
            "src/libllama.a",
            "common/libllama-common.a",
            "common/libllama-common-base.a",
            "ggml/src/libggml.a",
            "ggml/src/libggml-base.a",
            "ggml/src/ggml-cpu/libggml-cpu.a",
            "tools/mtmd/libmtmd.a",
            "vendor/hash/libvendor-hash.a",
        ):
            if relative == omit_archive:
                continue
            path = build_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
        stamp = build_dir / ".mesh-llm-build-stamp"
        stamp.write_text(
            "\n".join(
                (
                    "stamp-version=3",
                    "patched-sha=0123456789abcdef",
                    "backend=cpu",
                    f"link-mode={link_mode}",
                    f"toolchain-epoch={toolchain_epoch}",
                    "build-type=Release",
                    "cmake-arg=-DGGML_NATIVE=OFF",
                    "cmake-arg=-DGGML_OPENMP=OFF",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (build_dir / "CMakeCache.txt").write_text(
            "# Portable MeshLLM static ABI link metadata\n"
            "GGML_OPENMP_ENABLED:BOOL=OFF\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 3,
            "contract": "mesh-llm-static-abi-v3",
            "target_triple": manifest_target or target,
            "backend": "cpu",
            "build_directory": build_dir.name,
            "toolchain_epoch": toolchain_epoch,
            "build_stamp_sha256": sha256(stamp),
        }
        (build_dir / ".mesh-llm-static-abi-input.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        download = root / "download"
        download.mkdir()
        archive = download / "mesh-llm-static-abi.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(build_dir, arcname=build_dir.name)
            if sibling:
                sibling_file = root / "unexpected.txt"
                sibling_file.write_text("unexpected", encoding="utf-8")
                bundle.add(sibling_file, arcname=sibling_file.name)
        archive.with_name(f"{archive.name}.sha256").write_text(
            f"{sha256(archive)}  {archive.name}\n",
            encoding="utf-8",
        )
        return download

    def restore(
        self,
        download: Path,
        destination: Path,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["MESH_LLM_LLAMA_TOOLCHAIN_EPOCH"] = TOOLCHAIN_EPOCH
        return subprocess.run(
            [
                bash_executable(),
                RESTORE.as_posix(),
                download.as_posix(),
                destination.as_posix(),
                native_linux_target(),
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_restore_accepts_exact_typed_static_abi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = self.write_artifact(root)
            destination = root / "restored" / "build-stage-abi-static"

            result = self.restore(download, destination)

            self.assertEqual(
                result.returncode,
                0,
                result.stdout + result.stderr,
            )
            self.assertTrue((destination / "src/libllama.a").is_file())

    def test_stamp_verifier_allows_repeated_cmake_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = self.write_artifact(root)
            with tarfile.open(
                download / "mesh-llm-static-abi.tar.gz",
                "r:gz",
            ) as bundle:
                stamp = bundle.extractfile(
                    "build-stage-abi-static/.mesh-llm-build-stamp",
                )
                self.assertIsNotNone(stamp)
                stamp_path = root / "build-stamp"
                stamp_path.write_bytes(stamp.read())

            result = subprocess.run(
                [
                    sys.executable,
                    str(STAMP_VERIFIER),
                    str(stamp_path),
                    "--backend",
                    "cpu",
                    "--link-mode",
                    "static",
                    "--stamp-version",
                    "3",
                    "--toolchain-epoch",
                    TOOLCHAIN_EPOCH,
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cmake_arguments=2", result.stdout)

    def test_restore_rejects_manifest_target_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = self.write_artifact(
                root,
                manifest_target="mismatched-target",
            )

            result = self.restore(
                download,
                root / "restored" / "build-stage-abi-static",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("target_triple mismatch", result.stderr)

    def test_restore_rejects_non_static_build_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = self.write_artifact(root, link_mode="dynamic")

            result = self.restore(
                download,
                root / "restored" / "build-stage-abi-static",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("link-mode mismatch", result.stderr)

    def test_restore_rejects_incomplete_link_closure(self) -> None:
        # Every required archive is load-bearing for the link, so omitting any
        # one of them must fail the restore. vendor-hash is listed explicitly:
        # it was added after the others and only the Rust link line consumes
        # it, so a silent drop reappears as an undefined hash_sha256_hex far
        # from here.
        for omitted in (
            "common/libllama-common-base.a",
            "vendor/hash/libvendor-hash.a",
        ):
            with self.subTest(omitted=omitted):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    download = self.write_artifact(root, omit_archive=omitted)

                    result = self.restore(
                        download,
                        root / "restored" / "build-stage-abi-static",
                    )

                    self.assertNotEqual(result.returncode, 0)

    def test_restore_rejects_toolchain_epoch_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = self.write_artifact(
                root,
                toolchain_epoch="different-runner-image",
            )

            result = self.restore(
                download,
                root / "restored" / "build-stage-abi-static",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("toolchain_epoch mismatch", result.stderr)

    def test_restore_rejects_archive_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = self.write_artifact(root, sibling=True)

            result = self.restore(
                download,
                root / "restored" / "build-stage-abi-static",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "must contain exactly build-stage-abi-static",
                result.stderr,
            )

    def test_restore_rejects_extra_upload_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = self.write_artifact(root)
            (download / "unexpected.txt").write_text(
                "unexpected",
                encoding="utf-8",
            )

            result = self.restore(
                download,
                root / "restored" / "build-stage-abi-static",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "exactly its archive and checksum",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
