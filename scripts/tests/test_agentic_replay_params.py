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
        self.replay = json.loads(
            (ROOT / "ci/agentic-replay-nightly/matrix.json").read_text()
        )["replay"]

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
            [
                "all",
                "16",
                "2",
                "131072",
                "32768",
                "32768",
                "131072",
                "5",
                "2",
                "4",
                "2048",
                "1,2,4,8",
            ],
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
                ["AGENTIC_REPLAY_MODE=all"]
                + [
                    f"AGENTIC_REPLAY_{key.upper()}={self.replay[key]}"
                    for key in (
                        "sessions_per_concurrency",
                        "minimum_worker_waves",
                        "minimum_context_tokens",
                        "minimum_session_prompt_tokens",
                        "min_isl",
                        "max_isl",
                        "min_turns",
                        "passes",
                        "warmup_turns",
                        "max_output_tokens",
                    )
                ]
                + ["AGENTIC_REPLAY_CONCURRENCY=1,2,4,8"],
            )

    def test_rejects_invalid_replay_shapes(self) -> None:
        invalid_cases = {
            "checkpoint mode": {**self.replay, "mode": "checkpoint"},
            "final mode": {**self.replay, "mode": "final"},
            "insufficient waves": {**self.replay, "sessions_per_concurrency": 8},
            "short context": {**self.replay, "minimum_context_tokens": 32768},
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
