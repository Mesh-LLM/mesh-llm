"""CPU producer contract shared by unchanged and changed-pin canaries."""
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "scripts/skippy-workload-oracles-build.sh"


class WorkloadOracleProducerTests(unittest.TestCase):
    def test_environment_is_deterministic_and_does_not_override_metal_outputs(self) -> None:
        """The CPU oracle closure must not overwrite the primary Metal family outputs."""
        result = subprocess.run(
            ["bash", str(PRODUCER), "--print-env", "/tmp/canary with spaces"],
            text=True, capture_output=True, check=True,
        )
        values = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(6, len(values))
        self.assertTrue(all(name.startswith("SKIPPY_WORKLOAD_") for name in values))
        for suffix, executable in [("SERVER", "llama-server"), ("COMPLETION", "llama-completion"), ("TTS", "llama-tts")]:
            self.assertEqual(f"/tmp/canary with spaces/native/bin/{executable}", values[f"SKIPPY_WORKLOAD_ORACLE_{suffix}"])
        self.assertEqual("/tmp/canary with spaces/cargo/debug", values["SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR"])

    def test_rejects_relative_or_environment_injection_paths(self) -> None:
        """Only absolute, safely exportable producer paths can enter the generated environment."""
        for path in ["relative", "/tmp/line\nGH_TOKEN=bad", "/tmp/line\rnext"]:
            result = subprocess.run(["bash", str(PRODUCER), "--print-env", path], capture_output=True, check=False)
            self.assertNotEqual(0, result.returncode)

    def test_both_canary_paths_build_then_export_the_same_cpu_producers(self) -> None:
        """Normal and independent verification must consume the same explicit producer graph."""
        workflow = (ROOT / ".github/workflows/llama-upstream-canary.yml").read_text()
        repair = (ROOT / "scripts/llama-canary-agent-repair.sh").read_text()
        for text in [workflow, repair]:
            self.assertIn("just skippy-workload-oracles-build", text)
            self.assertIn("skippy-workload-oracles-build.sh --print-env", text)
        self.assertLess(workflow.index("name: Build pinned CPU workload"), workflow.index("name: Supported-families certification battery"))
        producer = PRODUCER.read_text()
        self.assertIn("-p skippy-topology --bins", producer)
        for contract in ["LLAMA_STAGE_BACKEND=cpu", "LLAMA_STAGE_LINK_MODE=static", "LLAMA_STAGE_WORKLOAD_ORACLE=ON", "CARGO_TARGET_DIR=", "--no-run --message-format=json", "--write-producer", "--source-snapshot"]:
            self.assertIn(contract, producer)
        consumer = (ROOT / "scripts/skippy-workload-certify.sh").read_text()
        self.assertIn("--producer-manifest", consumer)
        self.assertIn('TEST_COMMAND=("$(jq -er', consumer)

    def test_full_roster_and_cpu_producer_include_non_chat_on_every_trigger(self) -> None:
        """Nightly, pin changes, and forced runs share all six certified workload classes."""
        result = subprocess.run(
            [str(ROOT / "scripts/plan-family-battery.py")],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        rows = [row for row in json.loads(result.stdout)["selected_models"] if row["class"] != "causal_generation"]
        self.assertEqual(6, len(rows))
        self.assertEqual({"embedding", "rerank", "encoder_decoder", "ocr", "speech_synthesis", "speech_recognition"}, {row["class"] for row in rows})
        self.assertTrue(all(row["profile"] == "workload-oracle" for row in rows))
        workflow = (ROOT / ".github/workflows/llama-upstream-canary.yml").read_text()
        producer_step = workflow.split("- name: Build pinned CPU workload oracles and candidate\n", 1)[1].split("      - name:", 1)[0]
        self.assertNotIn("cadence", producer_step)
        self.assertIn("steps.sha.outputs.certify == 'true'", producer_step)
