import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = ROOT / "scripts/ci-two-node-split-smoke.sh"
PROMPT_COUNTS = [644, 644, 788, 788, 916, 916]


def prefix_validator_source() -> str:
    script = SMOKE_SCRIPT.read_text()
    function = script[script.index("validate_prefix_responses() {") :]
    match = re.search(r"<<'PY'\n(?P<source>.*?)\nPY\n}", function, re.DOTALL)
    if match is None:
        raise AssertionError("could not extract split-prefix response validator")
    return match.group("source")


class TwoNodeSplitSmokeTests(unittest.TestCase):
    def run_validator(
        self, cached_counts: list[int]
    ) -> subprocess.CompletedProcess[str]:
        self.assertEqual(len(cached_counts), len(PROMPT_COUNTS))
        with tempfile.TemporaryDirectory() as directory:
            response_dir = Path(directory)
            for index, (prompt_tokens, cached_tokens) in enumerate(
                zip(PROMPT_COUNTS, cached_counts), start=1
            ):
                response = {
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                }
                (response_dir / f"response-{index}.json").write_text(
                    json.dumps(response)
                )

            return subprocess.run(
                [sys.executable, "-", directory, "6", "kv-recurrent"],
                input=prefix_validator_source(),
                text=True,
                capture_output=True,
                check=False,
            )

    def test_prefix_validator_exit_behavior(self):
        cases = {
            "second request misses": ([0, 0, 512, 640, 640, 768], 1),
            "recurrent checkpoint reuse grows": ([0, 512, 512, 640, 640, 768], 0),
            "later repeated request misses": ([0, 512, 512, 640, 640, 0], 1),
            "all follow-up requests miss": ([0, 0, 0, 0, 0, 0], 75),
        }

        for name, (cached_counts, expected_exit) in cases.items():
            with self.subTest(name=name):
                result = self.run_validator(cached_counts)
                self.assertEqual(
                    result.returncode,
                    expected_exit,
                    msg=f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
                )


if __name__ == "__main__":
    unittest.main()
