from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "qa-agent-tool-call-reliability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qa_agent_tool_call_reliability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class AgentToolCallReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = load_module()

    def test_normalizes_mesh_base_url_to_v1(self) -> None:
        self.assertEqual(
            self.harness.normalize_v1_base("http://127.0.0.1:9337"),
            "http://127.0.0.1:9337/v1",
        )
        self.assertEqual(
            self.harness.normalize_v1_base("http://127.0.0.1:9337/v1/"),
            "http://127.0.0.1:9337/v1",
        )

    def test_print_plan_does_not_require_network(self) -> None:
        plan = self.harness.build_plan(["auto", "mesh"], attempts=2)
        self.assertEqual(
            [probe.model for probe in plan],
            ["auto", "auto", "mesh", "mesh"],
        )
        rendered = self.harness.render_plan(plan, "http://127.0.0.1:9337/v1")
        decoded = json.loads(rendered)
        self.assertEqual(decoded["name"], "agent-tool-call-reliability")
        self.assertEqual(decoded["endpoint"], "http://127.0.0.1:9337/v1")
        self.assertEqual(decoded["checks"][0]["model"], "auto")
        self.assertEqual(decoded["checks"][3]["attempt"], 2)

    def test_extracts_valid_tool_call(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_fixture_fact",
                                    "arguments": json.dumps({"key": "codeword"}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        call = self.harness.extract_tool_call(response)
        self.assertEqual(call.call_id, "call_1")
        self.assertEqual(call.key, "codeword")

    def test_rejects_wrong_tool_call_shape(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_fixture_fact",
                                    "arguments": json.dumps({"key": "unknown"}),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "unsupported fixture key"):
            self.harness.extract_tool_call(response)

    def test_extracts_streamed_tool_call(self) -> None:
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_fixture_fact",
                                        "arguments": '{"key"',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ':"checksum"}'},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        call = self.harness.extract_stream_tool_call(chunks)
        self.assertEqual(call.call_id, "call_stream")
        self.assertEqual(call.key, "checksum")

    def test_validates_tool_result_continuation(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "The fixture value is signal-7429.",
                    }
                }
            ]
        }
        self.harness.validate_final_answer(response, "signal-7429")

        bad_response = {"choices": [{"message": {"content": "no fixture here"}}]}
        with self.assertRaisesRegex(ValueError, "missing expected fixture value"):
            self.harness.validate_final_answer(bad_response, "signal-7429")

    def test_writes_jsonl_probe_results(self) -> None:
        result = self.harness.ProbeResult(
            model="auto",
            attempt=1,
            phase="tool_call",
            ok=True,
            detail="tool call accepted",
            elapsed_ms=12,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            self.harness.write_jsonl(path, [result])
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(rows[0]["model"], "auto")
        self.assertEqual(rows[0]["phase"], "tool_call")
        self.assertTrue(rows[0]["ok"])


if __name__ == "__main__":
    unittest.main()
