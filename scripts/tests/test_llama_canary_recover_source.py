"""The failed canary build must retain source without certifying it."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "llama-canary-recover-source.py"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


class RecoveryTests(unittest.TestCase):
    def test_recovery_restores_staged_unstaged_and_new_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            git(root, "init", "-q")
            git(root, "config", "user.name", "Test")
            git(root, "config", "user.email", "test@example.com")
            (root / "tracked.txt").write_text("base\n")
            (root / "binary.dat").write_bytes(b"\0base\xff")
            (root / ".gitignore").write_text(".deps/\n")
            git(root, "add", ".")
            git(root, "commit", "-qm", "base")
            base = git(root, "rev-parse", "HEAD")

            (root / "tracked.txt").write_text("staged\n")
            git(root, "add", "tracked.txt")
            (root / "tracked.txt").write_text("final\n")
            (root / "binary.dat").write_bytes(b"\0repaired\xff")
            (root / "new source.txt").write_text("new\n")
            nested = root / ".deps" / "llama.cpp"
            nested.mkdir(parents=True)
            git(nested, "init", "-q")
            git(nested, "config", "user.name", "Test")
            git(nested, "config", "user.email", "test@example.com")
            (nested / "runtime.cpp").write_text("upstream\n")
            git(nested, "add", "runtime.cpp")
            git(nested, "commit", "-qm", "upstream")
            nested_base = git(nested, "rev-parse", "HEAD")
            (nested / "runtime.cpp").write_text("repair\n")
            (nested / "new.cpp").write_text("new repair\n")
            output = Path(directory) / "evidence"
            subprocess.run([sys.executable, str(SOURCE), str(root), str(output), base], check=True)

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(base, manifest["base"])
            self.assertFalse(manifest["verified"])
            self.assertEqual(nested_base, manifest["prepared_llama_cpp"]["base"])
            self.assertEqual(1, manifest["prepared_llama_cpp"]["untracked_files"])
            restored = Path(directory) / "restored"
            git(Path(directory), "clone", "-q", str(root), str(restored))
            subprocess.run(["git", "apply", "--binary", str(output / "tracked.patch")],
                           cwd=restored, check=True)
            with tarfile.open(output / "untracked.tar.gz") as archive:
                archive.extractall(restored, filter="data")
            for name in ("tracked.txt", "binary.dat", "new source.txt"):
                self.assertEqual((root / name).read_bytes(), (restored / name).read_bytes())
            self.assertIn(b"repair", (output / "prepared-llama-cpp" / "tracked.patch").read_bytes())
            with tarfile.open(output / "prepared-llama-cpp" / "untracked.tar.gz") as archive:
                self.assertEqual(["new.cpp"], archive.getnames())

    def test_changed_head_is_recorded_and_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git(root, "init", "-q")
            git(root, "config", "user.name", "Test")
            git(root, "config", "user.email", "test@example.com")
            (root / "file").write_text("base")
            git(root, "add", "file")
            git(root, "commit", "-qm", "base")
            base = git(root, "rev-parse", "HEAD")
            (root / "file").write_text("committed repair")
            git(root, "commit", "-qam", "agent committed unexpectedly")
            output = root / "out"
            subprocess.run([sys.executable, str(SOURCE), str(root), str(output), base], check=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotEqual(base, manifest["head"])
            self.assertIn(b"committed repair", (output / "tracked.patch").read_bytes())


if __name__ == "__main__":
    unittest.main()
