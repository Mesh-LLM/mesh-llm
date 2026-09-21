from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "skippy-workload-certify.sh"


class WorkloadCertifyContractTests(unittest.TestCase):
    def test_http_config_uses_unsplit_graph_runtime_contract(self) -> None:
        """Execute the real config generator with CPU/GPU and optional projector inputs."""
        source = RUNNER.read_text(encoding="utf-8")
        generator = source.split('python3 - "$CONFIG_PATH"', 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        for gpu_layers in (0, 99):
            for projector in ("", "/fixture/projector.gguf"):
                with self.subTest(gpu_layers=gpu_layers, projector=projector):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        config_path = Path(temp_dir) / "stage.json"
                        result = subprocess.run(
                            [sys.executable, "-", str(config_path), "fixture-model",
                             "/fixture/model.gguf", "a" * 64, "12", str(gpu_layers), projector],
                            input=generator, text=True, capture_output=True, check=False,
                        )
                        self.assertEqual(0, result.returncode, result.stderr)
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertEqual([], config["resident_tensor_names"])
                    self.assertEqual("", config["execution_contract"])
                    self.assertNotIn("filter_tensors_on_load", config)
                    self.assertEqual((0, 12, 1), (config["layer_start"], config["layer_end"], config["lane_count"]))
                    self.assertEqual("runtime-slice", config["load_mode"])
                    self.assertEqual("a" * 64, config["source_model_sha256"])
                    self.assertEqual(projector or None, config.get("projector_path"))
                    self.assertEqual({"backend_device": "CPU"} if gpu_layers == 0 else None, config["selected_device"])
                    for offload in ("kv_offload", "op_offload"):
                        self.assertEqual(False if gpu_layers == 0 else None, config[offload])

    def test_cpu_candidate_stamp_is_checked_after_build_without_weakening_reuse(self) -> None:
        """Fresh builds may create their stamp; stale or unbound reused outputs cannot pass."""
        source = RUNNER.read_text(encoding="utf-8")
        check = "require_pinned_cpu_candidate() {" + source.split("require_pinned_cpu_candidate() {", 1)[1].split('if [[ -n "$ORACLE_SERVER" ]]', 1)[0]
        build = source.split("# The canary explicitly produces", 1)[1].split('MEDIA_PATH=""', 1)[0]
        build = build[build.index('if [[ -n "$PRODUCER_MANIFEST" ]]'):]
        for initial, built, skip, manifest, expected in (
            ("", "current", 0, "", 0),
            ("stale", "current", 0, "", 0),
            ("", "stale", 0, "", 1),
            ("", "metal", 0, "", 1),
            ("current", "current", 1, "", 1),
            ("current", "current", 1, "manifest", 0),
            ("stale", "current", 1, "manifest", 1),
        ):
            with self.subTest(initial=initial, built=built, skip=skip, manifest=manifest):
                with tempfile.TemporaryDirectory() as temp_dir:
                    fixture = r'''
set -euo pipefail
ROOT=repo CANDIDATE_BIN_DIR=bin ORACLE_SERVER=oracle ORACLE_COMPLETION= ORACLE_TTS=
TEST_COMMAND=(test)
write_stamp() {
  printf 'patched-sha=%s\nbackend=%s\nlink-mode=static\ncmake-arg=-DGGML_METAL=OFF\n' \
    "$1" "$2" > "$CANDIDATE_BUILD_DIR/.mesh-llm-build-stamp"
}
python3() {
  case "$1" in
    */llama-oracle-source.py) echo current ;;
    */check-skippy-workload-candidate.py) echo checked ;;
    *) return 99 ;;
  esac
}
jq() { echo prebuilt-test; }
cargo() {
  echo built
  if [[ "$built" == metal ]]; then write_stamp current metal; else write_stamp "$built" cpu; fi
}
if [[ -n "$initial" ]]; then write_stamp "$initial" cpu; fi
'''
                    result = subprocess.run(
                        ["bash", "-c", fixture + check + build], text=True, capture_output=True, check=False,
                        env={**os.environ, "CANDIDATE_BUILD_DIR": temp_dir, "initial": initial, "built": built,
                             "SKIP_BUILD": str(skip), "PRODUCER_MANIFEST": manifest},
                    )
                self.assertEqual(expected, result.returncode, result.stdout + result.stderr)
                self.assertEqual(not skip, "built" in result.stdout)
                if not skip and expected == 0:
                    self.assertLess(result.stdout.index("built"), result.stdout.index("checked"))
                if skip and not manifest:
                    self.assertIn("requires a source-bound", result.stderr)
                elif expected:
                    self.assertIn("current pinned CPU llama.cpp build stamp", result.stderr)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Invoke the certification wrapper and retain its status and diagnostics for assertions."""
        return subprocess.run(
            [str(RUNNER), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_help_documents_the_typed_certification_inputs(self) -> None:
        """Keep the public wrapper usage aligned with its class-specific inputs."""
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
        self.assertIn("--startup-timeout-secs", result.stderr)

    def test_startup_deadline_rejects_invalid_values_before_execution(self) -> None:
        """Malformed or out-of-range budgets must not reach native startup."""
        for value in ("0", "-1", "1.5", "01", "86401", "abc"):
            with self.subTest(value=value):
                result = self._run("--startup-timeout-secs", value)
                self.assertEqual(1, result.returncode)
                self.assertIn("must be a positive integer", result.stderr)

    def test_missing_embedding_sdk_fails_before_model_execution(self) -> None:
        """Embedding certification cannot proceed without its required official SDK lane."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            model.touch()
            result = subprocess.run(
                [str(RUNNER), "--class", "embedding", "--lane", "embedding-smoke",
                 "--model-path", str(model), "--model-id", "fixture", "--work-dir", temp_dir,
                 "--skip-build"], cwd=ROOT, text=True, capture_output=True, check=False,
                env={**os.environ, "SKIPPY_WORKLOAD_SDK_PYTHON": str(Path(temp_dir) / "missing-python")},
            )
            self.assertEqual(1, result.returncode)
            self.assertIn("official openai-python SDK smoke requires", result.stderr)
            self.assertFalse((Path(temp_dir) / "workload-oracle-evidence.json").exists())

    def test_startup_timeout_must_be_positive(self) -> None:
        """A zero startup budget is a usage error rather than an immediate runtime failure."""
        result = self._run("--startup-timeout-secs", "0")
        self.assertEqual(1, result.returncode)
        self.assertIn("must be a positive integer", result.stderr)

    def test_embedding_sdk_smoke_cannot_be_skipped(self) -> None:
        """Keep SDK verification mandatory instead of accepting a skipped lane as success."""
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("official openai-python SDK smoke requires", runner)
        self.assertNotIn("SDK smoke skipped", runner)

    def test_certified_mode_rejects_missing_oracle_before_build(self) -> None:
        """Certified mode requires the independent reference before consuming build resources."""
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
        """The wrapper must reject classes outside its closed workload vocabulary."""
        result = self._run("--class", "guessed", "--lane", "guessed-equivalence")
        self.assertEqual(1, result.returncode)
        self.assertIn("unsupported model class: guessed", result.stderr)

    def test_lane_must_match_the_selected_class(self) -> None:
        """A mismatched lane label cannot produce evidence for a different workload."""
        result = self._run(
            "--class",
            "embedding",
            "--lane",
            "rerank-smoke",
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("expected embedding-smoke", result.stderr)

    def test_projector_classes_fail_closed_without_a_projector(self) -> None:
        """OCR and speech certification require explicit projector inputs."""
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
        """A model file or other nonexecutable cannot masquerade as the reference server."""
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
        """Speech synthesis requires the dedicated waveform reference, not an HTTP server."""
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
        """The waveform executable cannot certify embedding or other server workloads."""
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
        """Encoder-decoder certification must use the direct completion reference."""
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
