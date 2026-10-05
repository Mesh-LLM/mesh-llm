from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[3]
HELPER = runpy.run_path(REPO / "skippy/evals/serving_cli.py")
serve_args = HELPER["serve_args"]
SPEC = importlib.util.spec_from_file_location(
    "skippy_cache_production_bench", REPO / "skippy/evals/skippy-cache-production-bench.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


class ServingCliTests(unittest.TestCase):
    def test_unified_interface_is_preferred_for_any_baseline_and_mode(self) -> None:
        binary = Path("old-baseline-skippy")
        for binary_transport, worker_only, expected in (
            (False, False, ["serve"]),
            (True, False, ["serve", "--stage-transport", "binary"]),
            (True, True, ["serve", "--stage-transport", "binary", "--worker-only"]),
        ):
            with self.subTest(mode=expected), mock.patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ) as probe:
                self.assertEqual(
                    serve_args(binary, binary_transport=binary_transport, worker_only=worker_only),
                    expected,
                )
                probe.assert_called_once_with(
                    [str(binary), "serve", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )

    def test_legacy_interfaces_keep_their_implicit_modes(self) -> None:
        binary = Path("legacy-skippy-server")
        for binary_transport, worker_only, legacy in (
            (False, False, "serve-openai"),
            (True, False, "serve-binary"),
            (True, True, "serve-binary"),
        ):
            with self.subTest(mode=legacy, worker_only=worker_only), mock.patch.object(
                subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 2, stderr="unknown subcommand"),
                    subprocess.CompletedProcess([], 0),
                ],
            ) as probe:
                self.assertEqual(
                    serve_args(binary, binary_transport=binary_transport, worker_only=worker_only),
                    [legacy],
                )
                self.assertEqual([call.args[0] for call in probe.call_args_list], [
                    [str(binary), "serve", "--help"],
                    [str(binary), legacy, "--help"],
                ])

    def test_unsupported_or_unresponsive_binaries_fail_explicitly(self) -> None:
        binary = Path("unsupported-skippy")
        for outcome, detail in (
            (subprocess.CompletedProcess([], 2, stderr="unknown subcommand"), "exit 2"),
            (subprocess.TimeoutExpired([str(binary)], 5.0), "timed out after 5s"),
        ):
            with self.subTest(detail=detail), mock.patch.object(subprocess, "run") as probe:
                if isinstance(outcome, Exception):
                    probe.side_effect = outcome
                else:
                    probe.return_value = outcome
                with self.assertRaisesRegex(RuntimeError, detail) as raised:
                    serve_args(binary)
                self.assertIn(str(binary), str(raised.exception))
                self.assertIn("serve-openai", str(raised.exception))
                self.assertEqual(probe.call_count, 2)
                self.assertTrue(all(call.kwargs["timeout"] == 5.0 for call in probe.call_args_list))

    def test_production_launch_uses_resolution_and_preserves_tuning_for_both_labels(self) -> None:
        case = BENCH.Case("fixture", "dense", "fixture/model", Path("fixture.gguf"), "resident-kv", 2, 8)
        args = Namespace(
            runtime_lane_count=3,
            serving_ctx_size=768,
            llama_stage_build_dir=Path("native-build"),
            server_startup_timeout_secs=1,
        )
        for label, command in (("old", "serve"), ("new", "serve-openai")):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                binary = Path(f"{label}-binary")
                with (
                    mock.patch.object(BENCH, "serve_args", return_value=[command]) as resolve,
                    mock.patch.object(BENCH, "write_skippy_benchmark_config"),
                    mock.patch.object(BENCH, "free_port", return_value=9337),
                    mock.patch.object(BENCH, "wait_ready"),
                    mock.patch.object(BENCH.subprocess, "Popen") as launch,
                ):
                    BENCH.run_skippy_serving_path_sweep(label, binary, case, "hello", [], args, output)
                resolve.assert_called_once_with(binary)
                self.assertEqual(launch.call_args.args[0], [
                    str(binary), command,
                    "--config", str(output / f"{label}-stage.json"),
                    "--bind-addr", "127.0.0.1:9337",
                    "--generation-concurrency", "3",
                    "--telemetry-level", "debug",
                ])


if __name__ == "__main__":
    unittest.main()
