from __future__ import annotations

import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR = ROOT / "scripts" / "safe-extract-tar.py"
CHECKSUM_VERIFIER = ROOT / "scripts" / "verify-checksum-sidecar.py"


class SafeExtractTarTests(unittest.TestCase):
    def run_extract(
        self,
        archive: Path,
        destination: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(EXTRACTOR), str(archive), str(destination)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def write_archive(
        self,
        archive: Path,
        members: list[tuple[tarfile.TarInfo, bytes | None]],
    ) -> None:
        with tarfile.open(archive, "w:gz") as bundle:
            for member, payload in members:
                bundle.addfile(
                    member,
                    io.BytesIO(payload) if payload is not None else None,
                )

    def test_extracts_regular_files_with_executable_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "runtime.tar.gz"
            directory = tarfile.TarInfo("runtime")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            tool = tarfile.TarInfo("runtime/tool")
            payload = b"#!/bin/sh\nexit 0\n"
            tool.size = len(payload)
            tool.mode = 0o755
            self.write_archive(
                archive,
                [(directory, None), (tool, payload)],
            )

            output = root / "output"
            result = self.run_extract(archive, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            extracted = output / "runtime" / "tool"
            self.assertEqual(extracted.read_bytes(), payload)
            self.assertNotEqual(extracted.stat().st_mode & 0o111, 0)

    def test_accepts_the_root_directory_member_emitted_by_tar_dot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "product.tar.gz"
            archive_root = tarfile.TarInfo(".")
            archive_root.type = tarfile.DIRTYPE
            payload = b"product"
            product = tarfile.TarInfo("./mesh-llm")
            product.size = len(payload)
            product.mode = 0o755
            self.write_archive(
                archive,
                [(archive_root, None), (product, payload)],
            )

            output = root / "output"
            result = self.run_extract(archive, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "mesh-llm").read_bytes(), payload)

    def test_rejects_traversal_absolute_and_device_members(self) -> None:
        cases: list[tuple[str, tarfile.TarInfo]] = []
        traversal = tarfile.TarInfo("../escape")
        traversal.size = 1
        cases.append(("traversal", traversal))
        absolute = tarfile.TarInfo("/absolute")
        absolute.size = 1
        cases.append(("absolute", absolute))
        device = tarfile.TarInfo("runtime/device")
        device.type = tarfile.CHRTYPE
        cases.append(("device", device))

        for name, member in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                archive = root / "unsafe.tar.gz"
                payload = b"x" if member.isreg() else None
                self.write_archive(archive, [(member, payload)])

                result = self.run_extract(archive, root / "output")

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsafe or invalid tar archive", result.stderr)

    def test_rejects_escaping_link_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "unsafe-link.tar.gz"
            link = tarfile.TarInfo("runtime/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../escape"
            self.write_archive(archive, [(link, None)])

            result = self.run_extract(archive, root / "output")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe or invalid tar archive", result.stderr)

    def test_rejects_nonempty_destination_with_preexisting_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "runtime.tar.gz"
            payload = b"must stay inside the extraction root"
            member = tarfile.TarInfo("redirect/payload")
            member.size = len(payload)
            self.write_archive(archive, [(member, payload)])

            output = root / "output"
            output.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (output / "redirect").symlink_to(outside, target_is_directory=True)

            result = self.run_extract(archive, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("destination must be empty", result.stderr)
            self.assertFalse((outside / "payload").exists())


class ChecksumSidecarTests(unittest.TestCase):
    def run_verify(self, artifact: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(CHECKSUM_VERIFIER), str(artifact)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_requires_exact_canonical_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "runtime.tar.gz"
            artifact.write_bytes(b"runtime")

            missing = self.run_verify(artifact)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("sidecar is missing or empty", missing.stderr)

            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            sidecar = artifact.with_name(f"{artifact.name}.sha256")
            sidecar.write_text(
                f"{digest}  {artifact.name}\n",
                encoding="utf-8",
            )
            valid = self.run_verify(artifact)
            self.assertEqual(valid.returncode, 0, valid.stderr)

            sidecar.write_text(
                f"{digest}  wrong-name.tar.gz\n",
                encoding="utf-8",
            )
            wrong_name = self.run_verify(artifact)
            self.assertNotEqual(wrong_name.returncode, 0)
            self.assertIn("checksum sidecar names", wrong_name.stderr)

            sidecar.write_text(
                f"{digest}  {artifact.name}\n{digest}  {artifact.name}\n",
                encoding="utf-8",
            )
            duplicate = self.run_verify(artifact)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("exactly one canonical line", duplicate.stderr)

    def test_rejects_noncanonical_checksum_separators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "runtime.tar.gz"
            artifact.write_bytes(b"runtime")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            sidecar = artifact.with_name(f"{artifact.name}.sha256")

            for separator in (" ", "\t", " *"):
                with self.subTest(separator=repr(separator)):
                    sidecar.write_text(
                        f"{digest}{separator}{artifact.name}\n",
                        encoding="utf-8",
                    )
                    result = self.run_verify(artifact)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "must use '<sha256>  <archive-name>' format",
                        result.stderr,
                    )

    def test_rejects_surrounding_whitespace_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "runtime.tar.gz"
            artifact.write_bytes(b"runtime")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            sidecar = artifact.with_name(f"{artifact.name}.sha256")

            sidecars = (
                f" {digest}  {artifact.name}\n",
                f"{digest}  {artifact.name} \n",
                f"\n{digest}  {artifact.name}\n",
                f"{digest}  {artifact.name}\n\n",
                f"{digest.upper()}  {artifact.name}\n",
            )
            for contents in sidecars:
                with self.subTest(contents=repr(contents)):
                    sidecar.write_text(contents, encoding="utf-8")
                    result = self.run_verify(artifact)
                    self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
