import hashlib
import io
import json
import pathlib
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "generate-native-runtime-release-manifest.sh"
)


class GenerateNativeRuntimeReleaseManifestTests(unittest.TestCase):
    def create_archive(self, root: pathlib.Path) -> pathlib.Path:
        package_dir = root / "meshllm-native-runtime-linux-aarch64-cpu"
        package_dir.mkdir()
        library = package_dir / "lib" / "runtime.bin"
        library.parent.mkdir()
        library.write_bytes(b"test native runtime")
        library_digest = hashlib.sha256(library.read_bytes()).hexdigest()
        (package_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "id": "meshllm-native-runtime-linux-aarch64-cpu",
                        "mesh_version": "0.68.0",
                        "skippy_abi": "0.1.25",
                        "platform": {
                            "os": "linux",
                            "arch": "aarch64",
                            "target": "aarch64-unknown-linux-gnu",
                        },
                        "backend": {"kind": "cpu"},
                        "rank": 0,
                        "libraries": ["lib/runtime.bin"],
                        "files": {
                            "lib/runtime.bin": library_digest,
                        },
                    },
                    "build": {
                        "primary_library": "lib/runtime.bin",
                        "library_sha256": library_digest,
                    },
                }
            ),
            encoding="utf-8",
        )
        archive = root / "meshllm-native-runtime-linux-aarch64-cpu.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(package_dir, arcname=package_dir.name)
        return archive

    def write_sidecar(
        self,
        archive: pathlib.Path,
        *,
        digest: str | None = None,
    ) -> None:
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_name(f"{archive.name}.sha256").write_text(
            f"{digest or actual}  {archive.name}\n",
            encoding="utf-8",
        )

    def run_generator(
        self,
        archive: pathlib.Path,
        out: pathlib.Path,
        *,
        tag: str = "v0.68.0",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(SCRIPT),
                "--tag",
                tag,
                "--out",
                str(out),
                str(archive),
            ],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_generated_manifest_is_single_valid_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = self.create_archive(root)
            self.write_sidecar(archive)

            out = root / "native-runtimes.json"
            result = self.run_generator(archive, out)
            self.assertEqual(result.returncode, 0, result.stderr)

            with out.open(encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["mesh_version"], "0.68.0")
            self.assertEqual(len(manifest["artifacts"]), 1)

    def test_requires_valid_canonical_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = self.create_archive(root)
            out = root / "native-runtimes.json"

            missing = self.run_generator(archive, out)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("sidecar is missing or empty", missing.stderr)
            self.assertFalse(out.exists())

            self.write_sidecar(archive, digest="0" * 64)
            corrupt = self.run_generator(archive, out)
            self.assertNotEqual(corrupt.returncode, 0)
            self.assertIn("archive checksum mismatch", corrupt.stderr)
            self.assertFalse(out.exists())

    def test_rejects_traversing_archive_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = self.create_archive(root)

            package_dir = root / "meshllm-native-runtime-linux-aarch64-cpu"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(package_dir, arcname=package_dir.name)
                escaping = tarfile.TarInfo("../escaped")
                payload = b"must not escape"
                escaping.size = len(payload)
                tar.addfile(escaping, io.BytesIO(payload))
            self.write_sidecar(archive)

            out = root / "native-runtimes.json"
            result = self.run_generator(archive, out)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe or invalid tar archive", result.stderr)
            self.assertFalse((root / "escaped").exists())
            self.assertFalse(out.exists())

    def test_rejects_runtime_version_that_does_not_match_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = self.create_archive(root)
            self.write_sidecar(archive)

            out = root / "native-runtimes.json"
            result = self.run_generator(
                archive,
                out,
                tag="v0.69.0-rc1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "mesh_version 0.68.0 does not match "
                "release tag v0.69.0-rc1",
                result.stderr,
            )
            self.assertFalse(out.exists())

    def test_rejects_sibling_payload_outside_artifact_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = self.create_archive(root)
            package_dir = root / "meshllm-native-runtime-linux-aarch64-cpu"

            with tarfile.open(archive, "w:gz") as tar:
                tar.add(package_dir, arcname=package_dir.name)
                sibling = tarfile.TarInfo("unexpected.txt")
                payload = b"not part of the runtime artifact"
                sibling.size = len(payload)
                tar.addfile(sibling, io.BytesIO(payload))
            self.write_sidecar(archive)

            out = root / "native-runtimes.json"
            result = self.run_generator(archive, out)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "expected archive to contain one top-level artifact directory",
                result.stderr,
            )
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
