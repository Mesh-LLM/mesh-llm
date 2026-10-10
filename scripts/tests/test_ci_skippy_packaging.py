"""Packaging evidence must describe the exact composed and archived product."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qualify_ci_packaging", ROOT / "skippy/scripts/qualify-ci-packaging.py",
)
assert SPEC is not None and SPEC.loader is not None
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)
SOURCE = "a" * 40


class PackagingQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.product_dir = self.root / "product"
        runtime_dir = self.product_dir / "native-runtimes" / "test-runtime"
        runtime_dir.mkdir(parents=True)
        build = {
            "schema_version": 1, "product": "skippy", "source_sha": SOURCE,
            "product_version": "0.78.0", "runtime_release": "0.78.0",
            "skippy_abi": "0.1.66", "os": "linux", "architecture": "x86_64",
        }
        binary = self.product_dir / "skippy"
        binary.write_text(
            "#!/usr/bin/env python3\nimport json, sys\n"
            f"build = {build!r}\n"
            "if '--version' in sys.argv: print('skippy ' + build['product_version'])\n"
            "elif 'build-contract' in sys.argv: print(json.dumps(build))\n"
            "else: print(json.dumps([{'native_runtime_id': 'test-runtime', "
            "'release_version': build['runtime_release'], "
            "'path': sys.argv[sys.argv.index('--runtime-bundle') + 1]}]))\n",
            encoding="utf-8",
        )
        binary.chmod(0o755)
        (self.product_dir / "build-contract.json").write_text(json.dumps(build), encoding="utf-8")
        (self.product_dir / "host-imports.json").write_text(json.dumps({
            "binary": "skippy", "binary_sha256": packaging.contract.digest(binary),
            "policy": "mesh-llm-dynamic-host-v2", "rejected_imports": [],
            "imports": [], "format": "elf",
        }), encoding="utf-8")
        runtime_manifest = {
            "schema_version": 2,
            "runtime": {
                "id": "test-runtime", "release_version": "0.78.0", "skippy_abi": "0.1.66",
                "platform": {"target": "x86_64-unknown-linux-gnu", "os": "linux", "arch": "x86_64"},
                "backend": {"kind": "cpu"},
            },
            "build": {"backend": "cpu"},
        }
        (runtime_dir / "manifest.json").write_text(json.dumps(runtime_manifest), encoding="utf-8")
        (runtime_dir / "libllama.so").write_bytes(b"fixture native runtime")
        product = {
            "schema_version": 1, "contract": "skippy-product-v1", "source_sha": SOURCE,
            "target": "x86_64-unknown-linux-gnu", "backend": "cpu",
            "cli": {
                "path": "skippy", "sha256": packaging.contract.digest(binary),
                "build_contract_sha256": packaging.contract.digest(self.product_dir / "build-contract.json"),
                "host_imports_sha256": packaging.contract.digest(self.product_dir / "host-imports.json"),
            },
            "runtime": {
                "id": "test-runtime", "path": "native-runtimes/test-runtime",
                "sha256": packaging.contract.tree_digest(runtime_dir),
                "manifest_sha256": packaging.contract.digest(runtime_dir / "manifest.json"),
            },
        }
        (self.product_dir / "product-manifest.json").write_text(json.dumps(product), encoding="utf-8")
        self.archive = self.root / "product.tar.gz"
        self.write_archive()

    def write_archive(self) -> None:
        with tarfile.open(self.archive, "w:gz") as package:
            for item in sorted(self.product_dir.rglob("*")):
                package.add(item, arcname=item.relative_to(self.product_dir).as_posix(), recursive=False)

    def test_exact_product_produces_packaging_evidence(self) -> None:
        evidence = packaging.verify(self.product_dir, self.archive, source_sha=SOURCE, row_id="linux-cpu")
        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["models"], {})
        self.assertEqual(set(evidence["executed_cases"]), packaging.contract.SUITE_CASES["packaging-runtime"])
        self.assertEqual(evidence["observations"]["runtime_id"], "test-runtime")

    def test_modified_archive_and_wrong_row_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected row"):
            packaging.verify(self.product_dir, self.archive, source_sha=SOURCE, row_id="linux-cuda")
        with tarfile.open(self.archive, "w:gz") as package:
            package.add(self.product_dir / "skippy", arcname="skippy")
        with self.assertRaisesRegex(ValueError, "archive file census"):
            packaging.verify(self.product_dir, self.archive, source_sha=SOURCE, row_id="linux-cpu")


if __name__ == "__main__":
    unittest.main()
