from __future__ import annotations

import argparse
from array import array
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "skippy-tts-oracle.py"
SPEC = importlib.util.spec_from_file_location("skippy_tts_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def write_wav(path: Path, samples: list[int], *, rate: int = 8000) -> None:
    pcm = array("h", samples)
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


class TtsOracleTests(unittest.TestCase):
    def test_prebuilt_candidate_requires_verified_producer_and_never_runs_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "producer.json"
            test_binary = str(root / "candidate tests")
            manifest.write_text(json.dumps({"files": {"test_binary": {"path": test_binary}}}))
            env = {
                "SKIPPY_WORKLOAD_PRODUCER_MANIFEST": str(manifest),
                "SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR": str(root / "bin"),
                "SKIPPY_WORKLOAD_NATIVE_BUILD_DIR": str(root / "native"),
            }
            with mock.patch.object(oracle.subprocess, "run") as verify:
                command = oracle.candidate_test_command(env)
            self.assertEqual([test_binary, oracle.TEST_NAME, "--exact", "--nocapture", "--test-threads=1"], command)
            verify.assert_called_once_with(
                [sys.executable, str(ROOT / "scripts/check-skippy-workload-candidate.py"),
                 "--candidate-binary", str(root / "bin/skippy-server"),
                 "--native-build-dir", str(root / "native"), "--producer-manifest", str(manifest)],
                cwd=ROOT, env=env, check=True,
            )
            with mock.patch.object(oracle.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["verify"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    oracle.candidate_test_command(env)

    def test_incomplete_producer_paths_do_not_fall_back_to_cargo(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires workload candidate and native build paths"):
            oracle.candidate_test_command({"SKIPPY_WORKLOAD_PRODUCER_MANIFEST": "producer.json"})

    def test_standalone_candidate_retains_explicit_cargo_test(self) -> None:
        command = oracle.candidate_test_command({})
        self.assertEqual(["cargo", "test"], command[:2])
        self.assertIn(oracle.TEST_NAME, command)

    def test_oracle_invocation_matches_candidate_no_repack_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_root = Path(temp_dir)
            model = test_root / "model.gguf"
            projector = test_root / "projector.gguf"
            model.write_bytes(b"fixture")
            projector.write_bytes(b"fixture")
            work_dir = test_root / "evidence"
            commands: list[list[str]] = []

            def fake_run_logged(command: list[str], _log_path: Path, **_kwargs: object) -> None:
                commands.append(command)
                name = "tts-candidate.wav" if command[0] == "cargo" else "tts-monolithic-oracle.wav"
                write_wav(work_dir / name, [1000, -1000] * 800)

            args = argparse.Namespace(
                oracle_cli=str(test_root / "bin" / "llama-tts"),
                model_path=str(model),
                projector_path=str(projector),
                model="qwen3-tts",
                layer_end=28,
                work_dir=str(work_dir),
            )
            with mock.patch.dict(os.environ, {"LLAMA_STAGE_BUILD_DIR": str(test_root / "abi")}, clear=True):
                with mock.patch.object(oracle, "require_pinned_cpu_oracle", return_value="pinned-sha"):
                    with mock.patch.object(oracle, "require_candidate_cpu_static_build"):
                        with mock.patch.object(oracle, "run_logged", side_effect=fake_run_logged):
                            result = oracle.run_oracle(args)

            self.assertEqual("pass", result["status"])
            self.assertEqual(2, len(commands))
            self.assertIn("--no-repack", commands[1])
            self.assertEqual("0", commands[1][commands[1].index("--min-p") + 1])
            self.assertEqual("0", commands[1][commands[1].index("-ngl") + 1])

    def test_identical_pcm_passes_with_zero_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.wav"
            reference = Path(temp_dir) / "reference.wav"
            samples = [1000 if index % 2 else -1000 for index in range(1600)]
            write_wav(candidate, samples)
            write_wav(reference, samples)
            metrics = oracle.compare_wavs(candidate, reference)
        self.assertEqual(0.0, metrics["relative_rms_error"])
        self.assertEqual(1.0, metrics["waveform_cosine"])
        self.assertEqual(1600, metrics["sample_count"])

    def test_small_pcm_rounding_difference_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.wav"
            reference = Path(temp_dir) / "reference.wav"
            write_wav(candidate, [1001, -999] * 800)
            write_wav(reference, [1000, -1000] * 800)
            metrics = oracle.compare_wavs(candidate, reference)
        self.assertLess(metrics["relative_rms_error"], oracle.MAX_RELATIVE_RMS_ERROR)

    def test_wrong_gain_and_phase_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.wav"
            reference = Path(temp_dir) / "reference.wav"
            write_wav(reference, [1000, -1000] * 800)
            for changed in ([1060, -1060] * 800, [-1000, 1000] * 800):
                with self.subTest(changed=changed[:2]):
                    write_wav(candidate, changed)
                    with self.assertRaisesRegex(RuntimeError, "differs from monolithic oracle"):
                        oracle.compare_wavs(candidate, reference)

    def test_silent_and_mismatched_wav_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.wav"
            reference = Path(temp_dir) / "reference.wav"
            write_wav(reference, [1000, -1000] * 800)
            write_wav(candidate, [0] * 1600)
            with self.assertRaisesRegex(RuntimeError, "silent"):
                oracle.compare_wavs(candidate, reference)
            write_wav(candidate, [1000, -1000] * 799)
            with self.assertRaisesRegex(RuntimeError, "sample count differs"):
                oracle.compare_wavs(candidate, reference)
            write_wav(candidate, [1000, -1000] * 800, rate=16000)
            with self.assertRaisesRegex(RuntimeError, "sample rate"):
                oracle.compare_wavs(candidate, reference)

    def test_oracle_requires_current_pinned_cpu_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_root = Path(temp_dir)
            binary = test_root / "build" / "bin" / "llama-tts"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"binary")
            binary.chmod(0o755)
            patched_sha = test_root / ".deps" / "llama.cpp" / ".mesh-llm-patched-sha"
            patched_sha.parent.mkdir(parents=True)
            patched_sha.write_text("pinned-sha\n", encoding="utf-8")
            stamp = test_root / "build" / ".mesh-llm-build-stamp"
            with mock.patch.object(oracle, "ROOT", test_root):
                with self.assertRaisesRegex(RuntimeError, "pinned build stamp"):
                    oracle.require_pinned_cpu_oracle(binary)
                stamp.write_text(
                    "patched-sha=pinned-sha\nbackend=cpu\nlink-mode=static\n"
                    "ggml-native=OFF\ncmake-arg=-DGGML_NATIVE=OFF\n"
                    "cmake-arg=-DGGML_METAL=OFF\n"
                    "cmake-arg=-DLLAMA_BUILD_TOOLS=ON\n",
                    encoding="utf-8",
                )
                oracle.require_pinned_cpu_oracle(binary)
                stamp.write_text(
                    "patched-sha=pinned-sha\nbackend=cpu\nlink-mode=static\n"
                    "ggml-native=ON\ncmake-arg=-DGGML_NATIVE=ON\n"
                    "cmake-arg=-DGGML_METAL=OFF\n"
                    "cmake-arg=-DLLAMA_BUILD_TOOLS=ON\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "pinned CPU build stamp"):
                    oracle.require_pinned_cpu_oracle(binary)
                stamp.write_text("patched-sha=pinned-sha\nbackend=metal\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "pinned CPU build stamp"):
                    oracle.require_pinned_cpu_oracle(binary)

    def test_candidate_requires_same_pinned_cpu_static_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir)
            stamp = build_dir / ".mesh-llm-build-stamp"
            with self.assertRaisesRegex(RuntimeError, "lacks a pinned static CPU build stamp"):
                oracle.require_candidate_cpu_static_build(build_dir, "pinned-sha")
            stamp.write_text(
                "patched-sha=old-sha\nbackend=cpu\nlink-mode=static\n"
                "ggml-native=OFF\ncmake-arg=-DGGML_NATIVE=OFF\n"
                "cmake-arg=-DGGML_METAL=OFF\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                oracle.require_candidate_cpu_static_build(build_dir, "pinned-sha")
            stamp.write_text(
                "patched-sha=pinned-sha\nbackend=cpu\nlink-mode=static\n"
                "ggml-native=OFF\ncmake-arg=-DGGML_NATIVE=OFF\n"
                "cmake-arg=-DGGML_METAL=OFF\n",
                encoding="utf-8",
            )
            oracle.require_candidate_cpu_static_build(build_dir, "pinned-sha")
            stamp.write_text(
                "patched-sha=pinned-sha\nbackend=cpu\nlink-mode=static\n"
                "ggml-native=ON\ncmake-arg=-DGGML_NATIVE=ON\n"
                "cmake-arg=-DGGML_METAL=OFF\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                oracle.require_candidate_cpu_static_build(build_dir, "pinned-sha")


if __name__ == "__main__":
    unittest.main()
