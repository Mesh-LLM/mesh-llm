#!/usr/bin/env python3
"""Regression tests for atomic, self-healing sccache downloads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "install-sccache.sh"
VERSION = "0.16.0"
ARCHIVE = f"sccache-v{VERSION}-x86_64-unknown-linux-musl.tar.gz"


class InstallSccacheTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        source = root / "source"
        payload = source / f"sccache-v{VERSION}-x86_64-unknown-linux-musl"
        payload.mkdir(parents=True)
        binary = payload / "sccache"
        binary.write_text("#!/bin/sh\necho 'sccache 0.16.0'\n", encoding="utf-8")
        binary.chmod(0o755)

        archive = source / ARCHIVE
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(payload, arcname=payload.name)
        checksum = source / f"{ARCHIVE}.sha256"
        checksum.write_text(
            f"{hashlib.sha256(archive.read_bytes()).hexdigest()}\n",
            encoding="utf-8",
        )
        return archive, checksum

    def make_curl(self, root: Path, source: Path, *, fail: bool = False) -> Path:
        fake_bin = root / "bin"
        fake_bin.mkdir(parents=True, exist_ok=True)
        curl = fake_bin / "curl"
        if fail:
            curl.write_text("#!/bin/sh\nexit 22\n", encoding="utf-8")
        else:
            curl.write_text(
                """#!/bin/sh
set -eu
url=''
destination=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) destination="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
cp "$SCCACHE_TEST_SOURCE/${url##*/}" "$destination"
""",
                encoding="utf-8",
            )
        curl.chmod(0o755)
        return fake_bin

    def run_installer(
        self, root: Path, source: Path, fake_bin: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(INSTALLER)],
            cwd=ROOT,
            env=os.environ
            | {
                "DOWNLOAD_CACHE_DIR": str(root / "cache"),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "SCCACHE_INSTALL_DIR": str(root / "install"),
                "SCCACHE_TEST_SOURCE": str(source),
                "SCCACHE_VERSION": VERSION,
                "TARGETARCH": "amd64",
            },
            text=True,
            capture_output=True,
            check=False,
        )

    def test_invalid_cache_is_replaced_before_atomic_promotion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mesh-install-sccache-") as temp:
            root = Path(temp)
            archive, checksum = self.make_fixture(root)
            cache = root / "cache"
            cache.mkdir()
            (cache / ARCHIVE).write_bytes(b"partial archive")
            (cache / f"{ARCHIVE}.sha256").write_text("invalid\n", encoding="utf-8")

            result = self.run_installer(root, archive.parent, self.make_curl(root, archive.parent))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((cache / ARCHIVE).read_bytes(), archive.read_bytes())
            self.assertEqual(
                (cache / f"{ARCHIVE}.sha256").read_text(encoding="utf-8"),
                checksum.read_text(encoding="utf-8"),
            )
            self.assertTrue((root / "install" / "sccache").is_file())
            self.assertEqual(list(cache.glob("*.tmp.*")), [])

            failing_bin = self.make_curl(root / "offline", archive.parent, fail=True)
            cached = self.run_installer(root, archive.parent, failing_bin)
            self.assertEqual(cached.returncode, 0, cached.stderr)

    def test_failed_refresh_leaves_no_partial_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mesh-install-sccache-failure-") as temp:
            root = Path(temp)
            archive, _ = self.make_fixture(root)
            cache = root / "cache"
            cache.mkdir()
            (cache / ARCHIVE).write_bytes(b"corrupt")

            result = self.run_installer(
                root, archive.parent, self.make_curl(root, archive.parent, fail=True)
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((cache / ARCHIVE).exists())
            self.assertFalse((cache / f"{ARCHIVE}.sha256").exists())
            self.assertEqual(list(cache.glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
