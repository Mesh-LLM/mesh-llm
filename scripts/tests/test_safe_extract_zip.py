from __future__ import annotations

import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "safe-extract-zip.py"


def symlink_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


class SafeExtractZipTests(unittest.TestCase):
    def run_extractor(
        self,
        archive: Path,
        destination: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(archive), str(destination)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_extracts_files_and_safe_framework_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.zip"
            destination = root / "output"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "MeshLLMFFI.xcframework/Info.plist",
                    "plist",
                )
                bundle.writestr(
                    "MeshLLMFFI.xcframework/"
                    "macos/MeshLLMFFI.framework/Versions/A/MeshLLMFFI",
                    "library",
                )
                bundle.writestr(
                    symlink_info(
                        "MeshLLMFFI.xcframework/"
                        "macos/MeshLLMFFI.framework/Versions/Current"
                    ),
                    "A",
                )
                bundle.writestr(
                    symlink_info(
                        "MeshLLMFFI.xcframework/"
                        "macos/MeshLLMFFI.framework/MeshLLMFFI"
                    ),
                    "Versions/Current/MeshLLMFFI",
                )

            result = self.run_extractor(archive, destination)

            self.assertEqual(result.returncode, 0, result.stderr)
            framework = (
                destination
                / "MeshLLMFFI.xcframework"
                / "macos"
                / "MeshLLMFFI.framework"
            )
            self.assertTrue((framework / "Versions/Current").is_symlink())
            self.assertEqual(
                (framework / "Versions/Current").readlink(),
                Path("A"),
            )
            self.assertEqual(
                (framework / "MeshLLMFFI").read_text(encoding="utf-8"),
                "library",
            )

    def test_rejects_entry_path_escapes(self) -> None:
        for name in ("../escape", "/absolute", r"C:\escape"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "input.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(name, "bad")

                result = self.run_extractor(archive, root / "output")

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsafe ZIP archive", result.stderr)
                self.assertFalse((root / "escape").exists())

    def test_rejects_escaping_or_ancestor_symlinks(self) -> None:
        fixtures = (
            (("link", "../../escape"), None),
            (("alias", "real"), ("alias/file", "bad")),
        )
        for link, nested in fixtures:
            with self.subTest(link=link), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "input.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(symlink_info(link[0]), link[1])
                    if nested is not None:
                        bundle.writestr(*nested)

                result = self.run_extractor(archive, root / "output")

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsafe ZIP archive", result.stderr)

    def test_rejects_duplicate_entries_and_nonempty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr("duplicate", "one")
                    bundle.writestr("duplicate", "two")

            duplicate = self.run_extractor(archive, root / "duplicates")
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("duplicate entry path", duplicate.stderr)

            clean_archive = root / "clean.zip"
            with zipfile.ZipFile(clean_archive, "w") as bundle:
                bundle.writestr("file", "content")
            destination = root / "nonempty"
            destination.mkdir()
            (destination / "sentinel").write_text("keep", encoding="utf-8")

            nonempty = self.run_extractor(clean_archive, destination)

            self.assertNotEqual(nonempty.returncode, 0)
            self.assertIn("destination must be empty", nonempty.stderr)
            self.assertEqual(
                (destination / "sentinel").read_text(encoding="utf-8"),
                "keep",
            )

            actual = root / "actual"
            actual.mkdir()
            symlink_destination = root / "symlink-destination"
            symlink_destination.symlink_to(actual, target_is_directory=True)
            symlinked = self.run_extractor(clean_archive, symlink_destination)
            self.assertNotEqual(symlinked.returncode, 0)
            self.assertIn("destination cannot be a symlink", symlinked.stderr)
            self.assertEqual(list(actual.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
