"""Terminal packaging receipts must bind the exact upstream release."""

import argparse
import importlib.util
import json
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "await-packaging-release.py"
spec = importlib.util.spec_from_file_location("await_packaging_release", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackagingReceiptTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            correlation_id="mesh-123-1-v0.78.0",
            upstream_repository="Mesh-LLM/mesh-llm",
            upstream_ref="v0.78.0",
            upstream_sha="a" * 40,
            manifest_sha256="b" * 64,
        )
        self.run = {"id": 456, "run_attempt": 1, "conclusion": "success"}
        self.receipt = {
            "schema": "mesh-packaging-readiness-v1",
            "status": "success",
            "correlation_id": self.args.correlation_id,
            "packaging_run_id": 456,
            "packaging_run_attempt": 1,
            "upstream": {
                "repository": self.args.upstream_repository,
                "ref": self.args.upstream_ref,
                "sha": self.args.upstream_sha,
                "manifest_sha256": self.args.manifest_sha256,
            },
            "requested": {
                "publish_images": "true",
                "publish_release_assets": "true",
                "publish_npm": "true",
            },
            "results": {"plan": "success"},
        }

    def test_accepts_correlated_terminal_success(self):
        module.validate_receipt(self.receipt, self.run, self.args)

    def test_rejects_foreign_manifest_and_failed_channel(self):
        self.receipt["upstream"]["manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "upstream release identity"):
            module.validate_receipt(self.receipt, self.run, self.args)
        self.receipt["upstream"]["manifest_sha256"] = self.args.manifest_sha256
        self.receipt["status"] = "failure"
        with self.assertRaisesRegex(ValueError, "did not complete"):
            module.validate_receipt(self.receipt, self.run, self.args)

    def test_rejects_wrong_attempt(self):
        self.receipt["packaging_run_attempt"] = 2
        with self.assertRaisesRegex(ValueError, "run identity"):
            module.validate_receipt(self.receipt, self.run, self.args)

    def test_rejects_missing_publication_channel(self):
        self.receipt["requested"]["publish_npm"] = "false"
        with self.assertRaisesRegex(ValueError, "required release channels"):
            module.validate_receipt(self.receipt, self.run, self.args)

    def test_rejects_duplicate_correlated_runs(self):
        run = {"display_title": f"Packaging · {self.args.correlation_id}"}
        payload = {"workflow_runs": [run, run]}
        with patch.object(module, "gh_api", return_value=json.dumps(payload).encode()):
            with self.assertRaisesRegex(ValueError, "more than one"):
                module.find_run("Mesh-LLM/mesh-packaging", self.args.correlation_id)

    def test_reads_only_exact_readiness_archive(self):
        payload = {"artifacts": [{"id": 9, "name": "packaging-readiness", "expired": False, "size_in_bytes": 100}]}
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w") as writer:
            writer.writestr("packaging-readiness.json", json.dumps(self.receipt))
        with patch.object(module, "gh_api", side_effect=[json.dumps(payload).encode(), archive.getvalue()]):
            self.assertEqual(module.load_receipt("Mesh-LLM/mesh-packaging", self.run), self.receipt)


if __name__ == "__main__":
    unittest.main()
