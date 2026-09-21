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
    """Hash fixture bytes using the same content identity recorded in evidence."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorkloadOracleEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create isolated model, producer and comparison evidence for each verifier case."""
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

    def run_verifier(self, model_class: str = "embedding", *extra: str) -> subprocess.CompletedProcess[str]:
        """Invoke the verifier against the selected class and independent fixture paths."""
        return subprocess.run(
            [
                "python3", str(VERIFIER), "--evidence", str(self.evidence),
                "--class", model_class, "--smoke-lane", "embedding-smoke",
                "--oracle-lane", "embedding-oracle", "--model-id", "fixture",
                "--model-path", str(self.model),
                "--candidate-executable", str(self.candidate),
                "--oracle-executable", str(self.oracle),
                "--pinned-patch-sha", "a" * 40,
                *extra,
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def run_writer(self, comparison: str, lane: str = "embedding-smoke", model_class: str = "embedding") -> subprocess.CompletedProcess[str]:
        """Run the evidence writer with a supplied comparison transcript and lane label."""
        comparison_log = Path(self.temp_dir.name) / "comparison.txt"
        comparison_log.write_text(comparison + "\n", encoding="utf-8")
        return subprocess.run(
            [
                "python3", str(WRITER), "--output", str(self.evidence),
                "--comparison-log", str(comparison_log), "--class", model_class,
                "--smoke-lane", lane, "--model-id", "fixture",
                "--model-sha256", sha256(self.model),
                "--candidate-executable", str(self.candidate),
                "--oracle-executable", str(self.oracle),
                "--pinned-patch-sha", "a" * 40,
                "--work-dir", self.temp_dir.name,
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def test_tts_writer_and_verifier_enforce_the_same_pcm_acceptance_bounds(self) -> None:
        """An asserted pass is insufficient without complete, finite, passing PCM metrics."""
        valid = {"sample_rate_hz": 24000, "channels": 1, "sample_count": 24000,
                 "relative_rms_error": 0.02, "waveform_cosine": 0.9995}
        projector = Path(self.temp_dir.name) / "projector.gguf"
        projector.write_bytes(b"projector")
        self.oracle = self.oracle.with_name("llama-tts")
        self.oracle.write_bytes(b"monolithic-tts")
        self.body.update({"class": "speech_synthesis", "oracle_executable": "llama-tts",
                          "oracle_executable_sha256": sha256(self.oracle),
                          "projector_sha256": sha256(projector),
                          "comparison": "speech_synthesis local-monolithic oracle passed: PCM comparison"})
        cases = [(valid, True), (None, False), ({}, False)]
        for field in valid:
            cases.append(({key: value for key, value in valid.items() if key != field}, False))
        for field in ("sample_rate_hz", "channels", "sample_count"):
            cases.extend(({**valid, field: value}, False) for value in (0, -1, True, 1.5, "1"))
        for field, invalid in (
            ("relative_rms_error", (-0.001, 0.020001, float("nan"), float("inf"), True, "0", 10 ** 999)),
            ("waveform_cosine", (0.99949, 1.0001, float("nan"), -float("inf"), False, "1")),
        ):
            cases.extend(({**valid, field: value}, False) for value in invalid)
        cases.append(({**valid, "relative_rms_error": 0, "waveform_cosine": 1}, True))
        result_path = Path(self.temp_dir.name) / "tts-oracle-result.json"
        for metrics, accepted in cases:
            with self.subTest(metrics=metrics):
                self.evidence.unlink(missing_ok=True)
                result_path.write_text(json.dumps({"status": "pass", "pinned_patch_sha": "a" * 40,
                                                   "metrics": metrics}), encoding="utf-8")
                written = self.run_writer(self.body["comparison"], model_class="speech_synthesis")
                self.assertEqual(0 if accepted else 1, written.returncode, written.stderr)
                self.assertEqual(accepted, self.evidence.exists())
                self.evidence.write_text(json.dumps({**self.body, "metrics": metrics}), encoding="utf-8")
                verified = self.run_verifier("speech_synthesis", "--projector-path", str(projector))
                self.assertEqual(0 if accepted else 1, verified.returncode, verified.stderr)
                if not accepted:
                    self.assertIn("TTS", written.stderr)
                    self.assertIn("TTS", verified.stderr)

    def test_comparator_pass_writes_verifiable_identity_bound_evidence(self) -> None:
        """A genuine comparator success produces evidence that passes independent verification."""
        written = self.run_writer(self.body["comparison"])
        self.assertEqual(0, written.returncode, written.stderr)
        self.assertEqual(0, self.run_verifier().returncode)

    def test_smoke_only_log_never_writes_oracle_evidence(self) -> None:
        """Protocol smoke output alone cannot be promoted to equivalence evidence."""
        written = self.run_writer("embedding OpenAI HTTP smoke passed")
        self.assertEqual(1, written.returncode)
        self.assertFalse(self.evidence.exists())

    def test_writer_rejects_missing_suffix_and_replaces_only_final_suffix(self) -> None:
        """Normalize only the terminal smoke suffix without changing embedded lane text."""
        for lane in ("embedding", "embedding-smoke-extra", "embedding-oracle"):
            with self.subTest(lane=lane):
                result = self.run_writer(self.body["comparison"], lane)
                self.assertEqual(1, result.returncode)
                self.assertIn("must end with '-smoke'", result.stderr)
                self.assertFalse(self.evidence.exists())
        result = self.run_writer(self.body["comparison"], "fixture-smoke-embedding-smoke")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("fixture-smoke-embedding-oracle", json.loads(self.evidence.read_text())["oracle_lane"])

    def test_projector_classes_require_independently_supplied_projector(self) -> None:
        """Projector-bearing workloads need an external sidecar identity to verify against."""
        for model_class in ("ocr", "speech_synthesis", "speech_recognition"):
            with self.subTest(model_class=model_class):
                self.body["class"] = model_class
                self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
                result = self.run_verifier(model_class)
                self.assertEqual(1, result.returncode)
                self.assertIn("requires a projector path", result.stderr)

    def test_projector_digest_is_verified_against_local_bytes(self) -> None:
        """Modified projector bytes must invalidate otherwise matching evidence."""
        projector = Path(self.temp_dir.name) / "projector.gguf"
        projector.write_bytes(b"projector")
        self.body.update({"class": "ocr", "projector_sha256": sha256(projector),
                          "comparison": "ocr local-monolithic oracle passed: exact text"})
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = self.run_verifier("ocr", "--projector-path", str(projector))
        self.assertEqual(0, result.returncode, result.stderr)
        projector.write_bytes(b"different projector")
        result = self.run_verifier("ocr", "--projector-path", str(projector))
        self.assertEqual(1, result.returncode)
        self.assertIn("projector_sha256 does not match", result.stderr)

    def test_matching_explicit_evidence_is_accepted(self) -> None:
        """Consistent explicit evidence remains accepted by the fail-closed verifier."""
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = self.run_verifier()
        self.assertEqual(0, result.returncode, result.stderr)

    def test_missing_evidence_is_rejected(self) -> None:
        """An absent evidence document cannot certify a completed-looking workload lane."""
        result = self.run_verifier()
        self.assertEqual(1, result.returncode)
        self.assertIn("workload oracle evidence rejected", result.stderr)

    def test_tampered_model_identity_is_rejected(self) -> None:
        """Changing the recorded model identity invalidates reference equivalence."""
        self.body["model_sha256"] = "b" * 64
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = self.run_verifier()
        self.assertEqual(1, result.returncode)
        self.assertIn("model_sha256 does not match", result.stderr)

    def test_smoke_only_output_cannot_certify_oracle_lane(self) -> None:
        """Reject evidence whose purported comparison is only a smoke transcript."""
        self.body["comparison"] = "embedding OpenAI HTTP smoke passed"
        self.evidence.write_text(json.dumps(self.body), encoding="utf-8")
        result = self.run_verifier()
        self.assertEqual(1, result.returncode)
        self.assertIn("lacks an explicit comparator pass", result.stderr)

    def test_writer_rejects_lane_without_smoke_suffix(self) -> None:
        """Reject malformed lane names before an oracle artifact can be written."""
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
        """A projector class cannot claim complete identity with only the main GGUF."""
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
