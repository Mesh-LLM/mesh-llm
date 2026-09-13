from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "skippy-workload-certify.sh"


class WorkloadCertifyContractTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(RUNNER), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_help_documents_the_typed_certification_inputs(self) -> None:
        result = self._run("--help")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--class CLASS", result.stderr)
        self.assertIn("--lane LANE", result.stderr)
        self.assertIn("--projector-path PATH", result.stderr)
        self.assertIn("--oracle-server PATH", result.stderr)
        self.assertIn("--oracle-completion PATH", result.stderr)
        self.assertIn("--oracle-tts PATH", result.stderr)
        self.assertIn("--startup-timeout-secs SECONDS", result.stderr)
        self.assertIn("--require-oracle", result.stderr)

    def test_startup_timeout_must_be_positive(self) -> None:
        result = self._run("--startup-timeout-secs", "0")
        self.assertEqual(1, result.returncode)
        self.assertIn("must be a positive integer", result.stderr)

    def test_embedding_sdk_smoke_cannot_be_skipped(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("official openai-python SDK smoke requires", runner)
        self.assertNotIn("SDK smoke skipped", runner)

    def test_certified_mode_rejects_missing_oracle_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            model.touch()
            result = self._run(
                "--class", "embedding", "--lane", "embedding-smoke",
                "--model-path", str(model), "--model-id", "fixture",
                "--work-dir", temp_dir, "--require-oracle", "--skip-build",
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("certified workload requires a class-appropriate", result.stderr)

    def test_unknown_class_fails_before_model_execution(self) -> None:
        result = self._run("--class", "guessed", "--lane", "guessed-equivalence")
        self.assertEqual(1, result.returncode)
        self.assertIn("unsupported model class: guessed", result.stderr)

    def test_lane_must_match_the_selected_class(self) -> None:
        result = self._run(
            "--class",
            "embedding",
            "--lane",
            "rerank-smoke",
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("expected embedding-smoke", result.stderr)

    def test_projector_classes_fail_closed_without_a_projector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            model.touch()
            for model_class, lane in (
                ("ocr", "ocr-smoke"),
                ("speech_synthesis", "speech-synthesis-smoke"),
                ("speech_recognition", "speech-recognition-smoke"),
            ):
                with self.subTest(model_class=model_class):
                    result = self._run(
                        "--class",
                        model_class,
                        "--lane",
                        lane,
                        "--model-path",
                        str(model),
                        "--model-id",
                        "fixture",
                        "--work-dir",
                        temp_dir,
                        "--skip-build",
                    )
                    self.assertEqual(1, result.returncode)
                    self.assertIn("requires a projector path", result.stderr)

    def test_oracle_requires_a_pinned_cpu_server_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            model.touch()
            result = self._run(
                "--class", "embedding", "--lane", "embedding-smoke",
                "--model-path", str(model), "--model-id", "fixture",
                "--work-dir", temp_dir, "--oracle-server", str(model),
                "--skip-build",
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("oracle executable is not executable", result.stderr)

    def test_speech_synthesis_rejects_server_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            projector = Path(temp_dir) / "projector.gguf"
            model.touch()
            projector.touch()
            result = self._run(
                "--class", "speech_synthesis", "--lane", "speech-synthesis-smoke",
                "--model-path", str(model), "--projector-path", str(projector),
                "--model-id", "fixture", "--work-dir", temp_dir,
                "--oracle-server", str(model), "--skip-build",
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("requires a different local-monolithic oracle", result.stderr)

    def test_other_classes_reject_tts_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            model.touch()
            result = self._run(
                "--class", "embedding", "--lane", "embedding-smoke",
                "--model-path", str(model), "--model-id", "fixture",
                "--work-dir", temp_dir, "--oracle-tts", str(model),
                "--skip-build",
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("only valid for speech synthesis", result.stderr)

    def test_encoder_decoder_requires_direct_completion_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            model.touch()
            result = self._run(
                "--class", "encoder_decoder", "--lane", "encoder-decoder-smoke",
                "--model-path", str(model), "--model-id", "fixture",
                "--work-dir", temp_dir, "--oracle-server", str(model),
                "--skip-build",
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("requires a different local-monolithic oracle", result.stderr)


if __name__ == "__main__":
    unittest.main()
