from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts" / "verify-workload-oracle-evidence.py"
WRITER = ROOT / "scripts" / "write-workload-oracle-evidence.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorkloadOracleEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.model = root / "model.gguf"
        self.model.write_bytes(b"pinned-model")
        self.candidate = root / "skippy-server"
        self.candidate.write_bytes(b"candidate")
        self.oracle = root / "llama-server"
        self.oracle.write_bytes(b"monolithic")
        self.evidence = root / "workload-oracle-evidence.json"
        self.body = {
            "status": "pass",
            "class": "embedding",
            "smoke_lane": "embedding-smoke",
            "oracle_lane": "embedding-oracle",
            "model_id": "fixture",
            "model_sha256": sha256(self.model),
            "projector_sha256": None,
            "candidate_executable_sha256": sha256(self.candidate),
            "oracle_executable": "llama-server",
            "oracle_executable_sha256": sha256(self.oracle),
            "pinned_patch_sha": "a" * 40,
            "comparison": "embedding local-monolithic oracle passed: max_abs_delta=0, min_cosine=1",
        }

    def run_verifier(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3", str(VERIFIER), "--evidence", str(self.evidence),
                "--class", "embedding", "--smoke-lane", "embedding-smoke",
                "--oracle-lane", "embedding-oracle", "--model-id", "fixture",
                "--model-path", str(self.model),
                "--candidate-executable", str(self.candidate),
                "--oracle-executable", str(self.oracle),
                "--pinned-patch-sha", "a" * 40,
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def run_writer(self, comparison: str) -> subprocess.CompletedProcess[str]:
        comparison_log = Path(self.temp_dir.name) / "comparison.txt"
        comparison_log.write_text(comparison + "\n", encoding="utf-8")
        return subprocess.run(
            [
                "python3", str(WRITER), "--output", str(self.evidence),
                "--comparison-log", str(comparison_log), "--class", "embedding",
                "--smoke-lane", "embedding-smoke", "--model-id", "fixture",
                "--model-sha256", sha256(self.model),
                "--candidate-executable", str(self.candidate),
                "--oracle-executable", str(self.oracle),
                "--pinned-patch-sha", "a" * 40,
                "--work-dir", self.temp_dir.name,
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def test_comparator_pass_writes_verifiable_identity_bound_evidence(self) -> None:
        written = self.run_writer(self.body["comparison"])
        self.assertEqual(0, written.returncode, written.stderr)
        self.assertEqual(0, self.run_verifier().returncode)

    def test_smoke_only_log_never_writes_oracle_evidence(self) -> None:
        written = self.run_writer("embedding OpenAI HTTP smoke passed")
        self.assertEqual(1, written.returncode)
        self.assertFalse(self.evidence.exists())

    def test_matching_explicit_evidence_is_accepted(self) -> None:
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = self.run_verifier()
        self.assertEqual(0, result.returncode, result.stderr)

    def test_missing_evidence_is_rejected(self) -> None:
        result = self.run_verifier()
        self.assertEqual(1, result.returncode)
        self.assertIn("workload oracle evidence rejected", result.stderr)

    def test_tampered_model_identity_is_rejected(self) -> None:
        self.body["model_sha256"] = "b" * 64
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = self.run_verifier()
        self.assertEqual(1, result.returncode)
        self.assertIn("model_sha256 does not match", result.stderr)

    def test_smoke_only_output_cannot_certify_oracle_lane(self) -> None:
        self.body["comparison"] = "embedding OpenAI HTTP smoke passed"
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = self.run_verifier()
        self.assertEqual(1, result.returncode)
        self.assertIn("lacks an explicit comparator pass", result.stderr)

    def test_writer_rejects_lane_without_smoke_suffix(self) -> None:
        comparison_log = Path(self.temp_dir.name) / "comparison.txt"
        comparison_log.write_text(self.body["comparison"] + "\n", encoding="utf-8")
        result = subprocess.run(
            [
                "python3", str(WRITER), "--output", str(self.evidence),
                "--comparison-log", str(comparison_log), "--class", "embedding",
                "--smoke-lane", "embedding-smoke-extra", "--model-id", "fixture",
                "--model-sha256", sha256(self.model),
                "--candidate-executable", str(self.candidate),
                "--oracle-executable", str(self.oracle),
                "--pinned-patch-sha", "a" * 40, "--work-dir", self.temp_dir.name,
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("must end with '-smoke'", result.stderr)

    def test_projector_workload_requires_projector_identity(self) -> None:
        self.body.update({
            "class": "ocr",
            "smoke_lane": "ocr-smoke",
            "oracle_lane": "ocr-oracle",
            "comparison": "ocr local-monolithic oracle passed: exact text",
        })
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = subprocess.run(
            [
                "python3", str(VERIFIER), "--evidence", str(self.evidence),
                "--class", "ocr", "--smoke-lane", "ocr-smoke",
                "--oracle-lane", "ocr-oracle", "--model-id", "fixture",
                "--model-path", str(self.model),
                "--candidate-executable", str(self.candidate),
                "--oracle-executable", str(self.oracle),
                "--pinned-patch-sha", "a" * 40,
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("requires a projector path", result.stderr)


if __name__ == "__main__":
    unittest.main()
