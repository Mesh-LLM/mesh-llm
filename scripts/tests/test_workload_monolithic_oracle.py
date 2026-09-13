from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci-workload-monolithic-oracle.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("workload_monolithic_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


class WorkloadMonolithicOracleTests(unittest.TestCase):
    def test_embedding_requires_dimension_and_numeric_parity(self) -> None:
        reference = {
            "data": [
                {"index": index, "embedding": [1.0, 0.0]}
                for index in range(len(oracle.EMBEDDING_INPUTS))
            ]
        }
        self.assertIn("max_abs_delta=0", oracle.compare_embeddings(reference, reference))
        changed = {
            "data": [
                {"index": index, "embedding": [0.9, 0.1]}
                for index in range(len(oracle.EMBEDDING_INPUTS))
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "differs from monolithic reference"):
            oracle.compare_embeddings(changed, reference)
        changed["data"][0]["embedding"] = [1.0]
        with self.assertRaisesRegex(RuntimeError, "dimensions differ"):
            oracle.compare_embeddings(changed, reference)

    def test_embedding_oracle_checks_each_single_input_after_batch(self) -> None:
        batch = {"data": [
            {"index": index, "embedding": [1.0, 0.0]}
            for index in range(len(oracle.EMBEDDING_INPUTS))
        ]}
        single = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        divergent = {"data": [{"index": 0, "embedding": [0.0, 1.0]}]}
        responses = [batch, batch, single, single, single, divergent, single, single]
        with patch.object(oracle, "request_json", side_effect=responses) as request:
            with self.assertRaisesRegex(RuntimeError, r"single\[1\]"):
                oracle.run_embedding_oracle("http://candidate", "http://oracle", "fixture")
        self.assertEqual(8, request.call_count)
        self.assertEqual(oracle.EMBEDDING_INPUTS[0], request.call_args_list[2].args[2]["input"])

    def test_rerank_requires_same_scores_and_order(self) -> None:
        reference = {"results": [
            {"index": 0, "relevance_score": 2.0},
            {"index": 1, "relevance_score": -1.0},
        ]}
        self.assertIn("max_abs_delta=0", oracle.compare_rerank(reference, reference))
        changed = {"results": [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.2},
        ]}
        with self.assertRaisesRegex(RuntimeError, "differs from monolithic reference"):
            oracle.compare_rerank(changed, reference)

    def test_encoder_decoder_compares_normalized_text(self) -> None:
        reference = {"choices": [{"text": "Das Haus ist wunderbar."}]}
        equivalent = {"choices": [{"text": " Das   Haus ist wunderbar.\n"}]}
        self.assertIn("identical normalized text", oracle.compare_encoder_decoder(equivalent, reference))
        changed = {"choices": [{"text": "Das Auto ist wunderbar."}]}
        with self.assertRaisesRegex(RuntimeError, "differs from monolithic reference"):
            oracle.compare_encoder_decoder(changed, reference)

    def test_direct_monolithic_completion_strips_only_terminal_runner_marker(self) -> None:
        result = subprocess.CompletedProcess(
            args=["llama-completion"], returncode=0,
            stdout=" Das Haus ist schön. [end of text]\n", stderr="model loaded",
        )
        with patch.object(oracle.subprocess, "run", return_value=result) as run:
            response = oracle.monolithic_completion("/bin/llama-completion", "/model.gguf")
        self.assertEqual("Das Haus ist schön.", response["choices"][0]["text"])
        self.assertIn("--no-display-prompt", run.call_args.args[0])
        self.assertIn("--temp", run.call_args.args[0])

    def test_direct_monolithic_completion_rejects_empty_or_failed_output(self) -> None:
        empty = subprocess.CompletedProcess(args=[], returncode=0,
                                            stdout=" [end of text]\n", stderr="")
        with patch.object(oracle.subprocess, "run", return_value=empty):
            with self.assertRaisesRegex(RuntimeError, "produced no text"):
                oracle.monolithic_completion("/bin/llama-completion", "/model.gguf")
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="bad")
        with patch.object(oracle.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "exited 1"):
                oracle.monolithic_completion("/bin/llama-completion", "/model.gguf")


if __name__ == "__main__":
    unittest.main()
