from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentic-replay-params.py"


class AgenticReplayParamsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.replay = {
            "mode": "checkpoint",
            "trajectories_per_framework": 8,
            "passes": 2,
            "warmup_turns": 4,
            "max_output_tokens": 2048,
            "concurrency": [1, 2, 4, 8],
            "dataset": "meshllm/example",
        }

    def run_script(
        self, replay: dict[str, object], *args: str
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            matrix_path = Path(temp_dir) / "matrix.json"
            matrix_path.write_text(json.dumps({"replay": replay}), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--matrix", str(matrix_path), *args],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_exports_validated_shell_values(self) -> None:
        result = self.run_script(self.replay, "--print-shell")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip().split("\t"),
            ["checkpoints", "8", "2", "4", "2048", "1,2,4,8"],
        )

    def test_writes_history_json_and_github_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            matrix_path = temp / "matrix.json"
            json_path = temp / "params.json"
            env_path = temp / "github.env"
            matrix_path.write_text(
                json.dumps({"replay": self.replay}), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--matrix",
                    str(matrix_path),
                    "--json-output",
                    str(json_path),
                    "--github-env",
                    str(env_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(json_path.read_text()), self.replay)
            self.assertEqual(
                env_path.read_text(encoding="utf-8").splitlines(),
                [
                    "AGENTIC_REPLAY_MODE=checkpoints",
                    "AGENTIC_REPLAY_TRAJECTORIES_PER_FRAMEWORK=8",
                    "AGENTIC_REPLAY_PASSES=2",
                    "AGENTIC_REPLAY_WARMUP_TURNS=4",
                    "AGENTIC_REPLAY_MAX_OUTPUT_TOKENS=2048",
                    "AGENTIC_REPLAY_CONCURRENCY=1,2,4,8",
                ],
            )

    def test_rejects_invalid_replay_shapes(self) -> None:
        invalid_cases = {
            "unknown mode": {**self.replay, "mode": "sometimes"},
            "boolean positive field": {**self.replay, "passes": True},
            "duplicate concurrency": {**self.replay, "concurrency": [1, 1]},
            "non-string mode": {**self.replay, "mode": ["checkpoint"]},
        }
        for name, replay in invalid_cases.items():
            with self.subTest(name=name):
                result = self.run_script(replay, "--print-shell")
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
