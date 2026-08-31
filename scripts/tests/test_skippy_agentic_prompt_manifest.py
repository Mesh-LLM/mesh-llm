from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def load_module():
    path = REPO / "evals/skippy-agentic-prompt-manifest.py"
    spec = importlib.util.spec_from_file_location("skippy_agentic_prompt_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MANIFEST = load_module()


class AgenticPromptManifestTest(unittest.TestCase):
    def test_flattens_roles_content_and_tool_calls(self) -> None:
        messages = [
            {"role": "system", "content": "rules", "tool_calls_json": None},
            {"role": "assistant", "content": "inspect", "tool_calls_json": '[{"id": 1}]'},
        ]

        flattened = MANIFEST.flatten_messages(json.dumps(messages))

        self.assertEqual(
            flattened,
            '<system>\nrules\n\n<assistant>\ninspect\n<tool_calls>[{"id": 1}]</tool_calls>',
        )

    def test_interleaves_repeated_trajectory_families(self) -> None:
        trajectories = [
            {
                "session_id": f"session-{index}",
                "source_dataset": "source",
                "messages_json": json.dumps([{"role": "user", "content": f"prefix-{index}"}]),
                "n_turns": 20,
                "max_isl": 9000,
                "total_tokens": 9100,
            }
            for index in range(2)
        ]

        manifest = MANIFEST.build_manifest(trajectories, 2, {"dataset_revision": "abc"})

        self.assertEqual(
            [prompt["family"] for prompt in manifest["prompts"]],
            ["trajectory-0", "trajectory-1", "trajectory-0", "trajectory-1"],
        )
        self.assertTrue(manifest["prompts"][2]["prompt"].startswith("<user>\nprefix-0"))
        self.assertEqual(manifest["metadata"]["rows"][1]["session_id"], "session-1")


if __name__ == "__main__":
    unittest.main()
