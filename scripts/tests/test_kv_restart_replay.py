from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evals/kv-restart-replay.py"


def load_module():
    spec = importlib.util.spec_from_file_location("kv_restart_replay", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


class KvRestartReplayTest(unittest.TestCase):
    def test_server_command_rejects_endpoint_overrides(self) -> None:
        binary = Path("/tmp/mesh-llm")

        with self.assertRaisesRegex(AssertionError, "--port"):
            BENCH.server_command(binary, "model.gguf", ["--port=9447"])
        with self.assertRaisesRegex(AssertionError, "--host"):
            BENCH.server_command(binary, "model.gguf", ["--host", "0.0.0.0"])

    def test_manifest_uses_distinct_deterministic_assistant_responses(self) -> None:
        manifest = BENCH.build_manifest(2, 32, 16)

        self.assertNotEqual(manifest["turns"][0]["request"], manifest["turns"][0]["response"])
        self.assertEqual(manifest, BENCH.build_manifest(2, 32, 16))

    def test_macos_memory_uses_sysctl(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="Apple M2\n"),
            subprocess.CompletedProcess([], 0, stdout="Mac14,6\n"),
            subprocess.CompletedProcess([], 0, stdout="17179869184\n"),
        ]
        with (
            mock.patch.object(BENCH.sys, "platform", "darwin"),
            mock.patch.object(BENCH.subprocess, "run", side_effect=results) as run,
        ):
            fingerprint = BENCH.hardware_fingerprint()

        self.assertEqual(fingerprint["physical_memory_bytes"], 17179869184)
        self.assertEqual(run.call_args_list[-1].args[0], ["sysctl", "-n", "hw.memsize"])

    def test_binary_provenance_keeps_unknown_source_sha_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "mesh-llm"
            binary.write_bytes(b"binary")
            with mock.patch.object(BENCH.subprocess, "run", side_effect=FileNotFoundError):
                provenance = BENCH.binary_provenance(binary)

        self.assertEqual(provenance["git_describe"], "unknown")
        self.assertEqual(provenance["source_sha"], "unknown")

    def test_stream_request_rejects_truncated_stream_with_usage(self) -> None:
        class TruncatedResponse:
            status = 200

            def __iter__(self):
                return iter(
                    [
                        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                        b'data: {"choices":[],"usage":{"completion_tokens":1,"prompt_tokens":10}}\n',
                    ]
                )

        class TruncatedConnection:
            def __init__(self, *_args, **_kwargs):
                pass

            def request(self, *_args, **_kwargs):
                pass

            def getresponse(self):
                return TruncatedResponse()

            def close(self):
                pass

        with mock.patch.object(BENCH.http.client, "HTTPConnection", TruncatedConnection):
            result = BENCH.stream_request(
                "request-1",
                [{"role": "user", "content": "task"}],
                "model",
                8,
                10,
            )

        self.assertEqual(result["error"], "stream ended without terminal [DONE] marker")

    def test_single_restore_sample_does_not_report_p95(self) -> None:
        summary = BENCH.summarize_cohort(
            "restore",
            [
                {
                    "ttft_seconds": 0.2,
                    "total_seconds": 0.3,
                    "prompt_tokens": 10,
                    "cached_tokens": 5,
                    "decode_tokens_per_second": 10.0,
                }
            ],
        )

        self.assertEqual(summary["ttft_p50_seconds"], 0.2)
        self.assertIsNone(summary["ttft_p95_seconds"])

    def test_run_arm_replays_the_last_fill_prompt_once_per_post_restart_request(self) -> None:
        calls: list[tuple[str, list[dict[str, str]]]] = []

        def fake_stream(request_id, messages, *_args):
            calls.append((request_id, list(messages)))
            return {
                "request_id": request_id,
                "ttft_seconds": 0.1,
                "total_seconds": 0.2,
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "cached_tokens": 5,
                "decode_tokens_per_second": 10.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "mesh-llm"
            model = root / "model.gguf"
            binary.write_bytes(b"binary")
            model.write_bytes(b"model")
            args = SimpleNamespace(
                binary=str(binary),
                model=str(model),
                turns=2,
                turn_target_tokens=32,
                system_tokens=16,
                restore_repeats=3,
                max_output_tokens=8,
                request_timeout=10.0,
                ready_timeout=10.0,
                serve_extra_args=[],
            )
            with (
                mock.patch.object(
                    BENCH,
                    "start_server",
                    side_effect=lambda *_args: (SimpleNamespace(), ["mesh-llm", "serve"]),
                ),
                mock.patch.object(BENCH, "stop_server"),
                mock.patch.object(BENCH, "wait_for_model", return_value="model"),
                mock.patch.object(BENCH, "stream_request", side_effect=fake_stream),
                mock.patch.object(BENCH, "binary_provenance", return_value={"source_sha": "a" * 40}),
                mock.patch.object(BENCH, "hardware_fingerprint", return_value={"platform": "test"}),
            ):
                run = BENCH.run_arm(args, root / "output")

        self.assertEqual([row["requests"] for row in run["cohorts"]], [2, 1, 2])
        self.assertEqual([request_id for request_id, _ in calls], [
            "fill-1",
            "fill-2",
            "restore-1",
            "warm-1",
            "warm-2",
        ])
        self.assertTrue(all(messages[-1]["role"] == "user" for _, messages in calls))
        self.assertEqual(calls[1][1], calls[2][1])
        self.assertEqual(calls[2][1], calls[3][1])


if __name__ == "__main__":
    unittest.main()
