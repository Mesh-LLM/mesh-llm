from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "skippy/evals/wan-lab/entrypoint.sh"


class WanLabEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.tools = self.root / "bin"
        self.tools.mkdir()
        self.launch_log = self.root / "launch.jsonl"
        self.helper_log = self.root / "helpers.log"
        self.config = self.root / "admitted stage.json"
        self.config_bytes = (
            '{\n  "model_id": "org/admitted-model", "lane_count": 2,\n'
            '  "layer_start": 12, "layer_end": 24,\n'
            '  "activation_codec": "bf16-rne-v1",\n'
            '  "source_model_sha256": "admitted-source",\n'
            '  "resident_tensor_names": ["blk.12.attn_q.weight"],\n'
            '  "execution_contract": "admitted-contract",\n'
            '  "activation_import_identities": ["frontier/layer-12"],\n'
            '  "activation_export_identities": ["frontier/layer-24"]\n}\n\n'
        ).encode()
        self.config.write_bytes(self.config_bytes)
        self.write_tool(
            "skippy",
            "import json, os, sys\n"
            "with open(os.environ['WAN_TEST_LAUNCH_LOG'], 'a') as handle:\n"
            "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1:2] != ['serve']:\n"
            "    raise SystemExit('unexpected admission or other skippy command')\n",
        )
        self.write_tool(
            "jq",
            "import json, sys\n"
            "expression, path = sys.argv[-2:]\n"
            "with open(path) as handle:\n"
            "    config = json.load(handle)\n"
            "fields = {'.model_id': 'model_id', '.lane_count': 'lane_count',\n"
            "          '.layer_start': 'layer_start', '.layer_end': 'layer_end',\n"
            "          '.activation_codec // \\\"raw-f32-v1\\\"': 'activation_codec'}\n"
            "if expression not in fields:\n"
            "    raise SystemExit('unexpected jq expression: ' + expression)\n"
            "print(config[fields[expression]])\n",
        )
        for name in ("skippy-package-builder", "tc", "curl"):
            self.write_tool(
                name,
                "import os, sys\n"
                "with open(os.environ['WAN_TEST_HELPER_LOG'], 'a') as handle:\n"
                "    handle.write(sys.argv[0] + '\\n')\n"
                "raise SystemExit('unexpected external helper invocation')\n",
            )

    def write_tool(self, name: str, body: str) -> None:
        path = self.tools / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(0o755)

    def run_stage(self, stage_index: int, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": f"{self.tools}{os.pathsep}{os.defpath}",
            "CONFIG_PATH": str(self.config),
            "STAGE_INDEX": str(stage_index),
            "STAGE_COUNT": "4",
            "WAN_ENABLE": "0",
            "WAN_TEST_LAUNCH_LOG": str(self.launch_log),
            "WAN_TEST_HELPER_LOG": str(self.helper_log),
        }
        env.update(overrides)
        return subprocess.run(
            ["/bin/bash", str(ENTRYPOINT), "stage"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def launched_args(self) -> list[str]:
        calls = [json.loads(line) for line in self.launch_log.read_text().splitlines()]
        self.assertEqual(len(calls), 1, calls)
        self.assertFalse(self.helper_log.exists())
        return calls[0]

    def base_args(self) -> list[str]:
        return [
            "serve", "--stage-transport", "binary",
            "--config", str(self.config),
            "--activation-codec", "bf16-rne-v1",
            "--metrics-otlp-grpc", "http://metrics:14317",
            "--telemetry-queue-capacity", "4096",
            "--telemetry-level", "debug",
            "--max-inflight", "2",
        ]

    def test_supplied_admitted_config_is_byte_preserved(self) -> None:
        result = self.run_stage(
            1,
            MODEL_PATH=str(self.root / "missing-model.gguf"),
            MODEL_PACKAGE_REF="hf://unused/package@main",
            CONFIG_DIR=str(self.root / "should-not-be-created"),
            N_BATCH="999",
            ACTIVATION_WIRE_DTYPE="f16",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.config.read_bytes(), self.config_bytes)
        args = self.launched_args()
        self.assertEqual(args[args.index("--config") + 1], str(self.config))
        self.assertEqual(args[args.index("--activation-codec") + 1], "f16-rne-v1")
        self.assertFalse((self.root / "should-not-be-created").exists())

    def test_head_launches_current_binary_transport_with_public_api(self) -> None:
        result = self.run_stage(
            0,
            OPENAI_BIND_ADDR="127.0.0.1:19437",
            OPENAI_DEFAULT_MAX_TOKENS="41",
            OPENAI_GENERATION_CONCURRENCY="3",
            OPENAI_PREFILL_CHUNK_SIZE="192",
            OPENAI_PREFILL_CHUNK_POLICY="fixed",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.launched_args(),
            self.base_args() + [
                "--bind-addr", "127.0.0.1:19437",
                "--model-id", "org/admitted-model",
                "--default-max-tokens", "41",
                "--generation-concurrency", "3",
                "--prefill-chunk-size", "192",
                "--prefill-chunk-policy", "fixed",
                "--prefill-adaptive-start", "128",
                "--prefill-adaptive-step", "128",
                "--prefill-adaptive-max", "512",
            ],
        )

    def test_worker_launches_current_binary_transport_without_public_api(self) -> None:
        result = self.run_stage(1, OPENAI_BIND_ADDR="127.0.0.1:19437")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.launched_args(), self.base_args() + ["--worker-only"])

    def test_unsupported_activation_dtype_is_rejected_before_launch(self) -> None:
        result = self.run_stage(0, ACTIVATION_WIRE_DTYPE="q8")
        self.assertEqual(result.returncode, 64, result.stderr)
        self.assertIn("unsupported ACTIVATION_WIRE_DTYPE: q8", result.stderr)
        self.assertFalse(self.launch_log.exists())
        self.assertFalse(self.helper_log.exists())
        self.assertEqual(self.config.read_bytes(), self.config_bytes)


if __name__ == "__main__":
    unittest.main()
