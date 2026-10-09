"""Fail closed when standalone producer identities cannot form one product."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "skippy/scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CLI = load("skippy_cli_input", "verify-cli-input.py")
PRODUCT = load("skippy_ci_product", "compose-ci-product.py")
SOURCE = "a" * 40


class StandaloneProductContractTests(unittest.TestCase):
    def test_windows_verifier_selects_git_bash_instead_of_wsl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Git"
            for subdir in ("cmd", "bin"):
                (root / subdir).mkdir(parents=True)
            git = root / "cmd" / "git.exe"
            git.touch()
            bash = root / "bin" / "bash.exe"
            bash.touch()
            with mock.patch.object(PRODUCT.shutil, "which", return_value=str(git)):
                self.assertEqual(PRODUCT.verification_bash(windows=True), str(bash.resolve()))
            bash.unlink()
            with mock.patch.object(PRODUCT.shutil, "which", return_value=str(git)):
                with self.assertRaisesRegex(FileNotFoundError, "Git Bash"):
                    PRODUCT.verification_bash(windows=True)

    def test_cli_rechecks_executable_checksum_report_and_embedded_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = {
                "schema_version": 1, "product": "skippy", "product_version": "0.78.0",
                "runtime_release": "0.78.0", "skippy_abi": "0.1.66",
                "source_sha": SOURCE, "os": "linux", "architecture": "x86_64",
            }
            binary = directory / "skippy"
            binary.write_text(
                "#!/usr/bin/env python3\nimport json, sys\n"
                "print('skippy 0.78.0' if '--version' in sys.argv else json.dumps(" + repr(contract) + "))\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            (directory / "skippy.sha256").write_text(f"{digest}  skippy\n", encoding="utf-8")
            (directory / "host-imports.json").write_text(json.dumps({
                "binary": "skippy", "binary_sha256": digest,
                "policy": "mesh-llm-dynamic-host-v2", "rejected_imports": [],
                "imports": [], "format": "elf",
            }), encoding="utf-8")
            (directory / "build-contract.json").write_text(json.dumps(contract), encoding="utf-8")
            binary.chmod(0o644)  # Downloaded Actions artifacts lose executable mode.
            self.assertEqual(CLI.verify(directory, SOURCE), contract)
            self.assertTrue(binary.stat().st_mode & 0o100)
            with self.assertRaisesRegex(ValueError, "source commit"):
                CLI.verify(directory, "b" * 40)
            (directory / "build-contract.json").write_text(json.dumps({**contract, "skippy_abi": "0.0.0"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from executable"):
                CLI.verify(directory, SOURCE)
            (directory / "build-contract.json").write_text(json.dumps(contract), encoding="utf-8")
            binary.write_bytes(binary.read_bytes() + b"# tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum sidecar"):
                CLI.verify(directory, SOURCE)

    def test_runtime_target_backend_release_and_abi_must_match_cli(self):
        cli = {
            "source_sha": SOURCE, "os": "linux", "architecture": "x86_64",
            "runtime_release": "0.78.0", "skippy_abi": "0.1.66",
        }
        manifest = {
            "schema_version": 2,
            "runtime": {
                "id": "native-test", "release_version": "0.78.0",
                "skippy_abi": "0.1.66",
                "platform": {"target": "x86_64-unknown-linux-gnu", "os": "linux", "arch": "x86_64"},
                "backend": {"kind": "cuda"},
            },
            "build": {"backend": "cuda"},
        }
        self.assertEqual(
            PRODUCT.validate_pair(cli, manifest, source_sha=SOURCE, target="x86_64-unknown-linux-gnu", backend="cuda")["id"],
            "native-test",
        )
        failures = (
            (cli, {**manifest, "runtime": {**manifest["runtime"], "skippy_abi": "0.0.0"}}, "ABI"),
            (cli, {**manifest, "runtime": {**manifest["runtime"], "release_version": "0.77.0"}}, "release"),
            (cli, {**manifest, "runtime": {**manifest["runtime"], "backend": {"kind": "cpu"}}}, "backend"),
            (cli, {**manifest, "runtime": {**manifest["runtime"], "platform": {"target": "aarch64-apple-darwin"}}}, "target"),
            ({**cli, "source_sha": "b" * 40}, manifest, "source"),
        )
        for bad_cli, bad_manifest, reason in failures:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                PRODUCT.validate_pair(bad_cli, bad_manifest, source_sha=SOURCE, target="x86_64-unknown-linux-gnu", backend="cuda")

    def test_runtime_producer_source_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                PRODUCT.verify_runtime_source(directory, SOURCE)
            (directory / "ci-source.json").write_text(json.dumps({"source_sha": "b" * 40}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "producer source"):
                PRODUCT.verify_runtime_source(directory, SOURCE)
            (directory / "ci-source.json").write_text(json.dumps({"source_sha": SOURCE}), encoding="utf-8")
            PRODUCT.verify_runtime_source(directory, SOURCE)

    def test_composed_binary_must_discover_selected_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime_dir = directory / "native-runtime"
            runtime_dir.mkdir()
            binary = directory / "skippy"
            binary.write_text(
                "#!/usr/bin/env python3\nimport json, sys\n"
                "print(json.dumps([{'native_runtime_id': 'native-test', "
                "'release_version': '0.78.0', "
                "'path': sys.argv[sys.argv.index('--runtime-bundle') + 1]}]))\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            runtime = {"id": "native-test", "release_version": "0.78.0"}
            PRODUCT.verify_discovery(binary, runtime_dir, runtime)
            with self.assertRaisesRegex(ValueError, "did not discover"):
                PRODUCT.verify_discovery(binary, runtime_dir, {**runtime, "id": "foreign"})


if __name__ == "__main__":
    unittest.main()
