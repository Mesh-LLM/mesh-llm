"""Two-node Auto selection certification rejects a wrong second stage."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/assert-split-stage-payloads.py"
SPEC = importlib.util.spec_from_file_location("split_payloads", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    evidence = {
        "status": "ready", "model_id": "pinned-model",
        "topology": {
            "model_id": "pinned-model", "run_id": "current-run",
            "topology_id": "topology", "stages": [
                {"stage_id": "stage-0", "stage_index": 0, "node_id": "seed-node-full"},
                {"stage_id": "stage-1", "stage_index": 1, "node_id": "worker-node-full"},
            ],
        },
        "observers": {
            "seed": {"node_id": "seed-node"},
            "worker": {"node_id": "worker-node"},
        },
    }
    selection = {
        "admitted_graph_state": "dense", "loaded_state_kind": "Dense",
        "resident": True, "kv_recurrent": True, "payload": "ResidentKv",
        "reason": "admitted_graph_and_exporter", "fallbacks": 0,
        "graph_loaded_state_mismatches": 0,
    }
    expectation = {
        "artifact_id": "pinned", "sha256": "a" * 64,
        "stages": [selection.copy(), selection.copy()],
    }
    def event(stage_index, *, payload="ResidentKv", run="current-run"):
        return {
            "event": "stage.kv_payload_selected",
            "attributes": {
                "skippy.run_id": run, "skippy.model_id": "pinned-model",
                "skippy.topology_id": "topology",
                "skippy.stage_id": f"stage-{stage_index}",
                "skippy.stage_index": stage_index,
                "skippy.kv.admitted_graph_state": "dense",
                "skippy.kv.loaded_state_kind": "Dense",
                "skippy.kv.loaded_memory_cache_resident": True,
                "skippy.kv.loaded_memory_cache_kv_recurrent": True,
                "skippy.kv.payload": payload,
                "skippy.kv.payload_selection_reason": "admitted_graph_and_exporter",
                "skippy.kv.payload_fallbacks": 0,
                "skippy.kv.graph_loaded_state_mismatches": 0,
            },
        }
    logs = {"seed": [event(0)], "worker": [event(1)]}
    return evidence, expectation, logs, event


class StagePayloadCertificationTests(unittest.TestCase):
    def test_pinned_expectations_match_smoke_manifest(self):
        expectations = json.loads((
            ROOT / "ci/model-artifacts/kv-auto-smoke-expectations.json"
        ).read_text())
        manifest = json.loads((
            ROOT / "ci/model-artifacts/manifests/scripted-binary-smoke.json"
        ).read_text())
        artifacts = {artifact["id"]: artifact for artifact in manifest["artifacts"]}
        for artifact_id, expected in expectations["models"].items():
            with self.subTest(artifact_id=artifact_id):
                self.assertEqual(expected["artifact_id"], artifact_id)
                self.assertEqual(expected["revision"], artifacts[artifact_id]["revision"])
                self.assertEqual(expected["sha256"], artifacts[artifact_id]["sha256"])

    def certify(self, evidence, expectation, logs):
        return MODULE.certify(evidence, expectation, logs, "pinned", "a" * 64)

    def test_both_stages_match(self):
        evidence, expectation, logs, _ = fixture()
        result = self.certify(evidence, expectation, logs)
        self.assertEqual([stage["stage_id"] for stage in result["stages"]], ["stage-0", "stage-1"])

    def test_wrong_second_stage_fails_even_when_first_matches(self):
        evidence, expectation, logs, event = fixture()
        logs["worker"] = [event(1, payload="FullState")]
        with self.assertRaisesRegex(MODULE.PayloadError, "stage stage-1 payload"):
            self.certify(evidence, expectation, logs)

    def test_explicit_mismatch_full_state_expectation(self):
        evidence, expectation, logs, _ = fixture()
        mismatch = expectation["stages"][1]
        mismatch.update({
            "loaded_state_kind": "Hybrid", "payload": "FullState",
            "reason": "graph_loaded_state_mismatch", "fallbacks": 1,
            "graph_loaded_state_mismatches": 1,
        })
        attrs = logs["worker"][0]["attributes"]
        attrs["skippy.kv.loaded_state_kind"] = "Hybrid"
        attrs["skippy.kv.payload"] = "FullState"
        attrs["skippy.kv.payload_selection_reason"] = "graph_loaded_state_mismatch"
        attrs["skippy.kv.payload_fallbacks"] = 1
        attrs["skippy.kv.graph_loaded_state_mismatches"] = 1
        self.assertEqual(self.certify(evidence, expectation, logs)["status"], "pass")

    def test_stale_event_does_not_mask_current_wrong_stage(self):
        evidence, expectation, logs, event = fixture()
        logs["worker"] = [event(1, run="old-run"), event(1, payload="FullState")]
        with self.assertRaisesRegex(MODULE.PayloadError, "stage stage-1 payload"):
            self.certify(evidence, expectation, logs)

    def test_missing_or_contradictory_selection_fails(self):
        evidence, expectation, logs, event = fixture()
        logs["worker"] = []
        with self.assertRaisesRegex(MODULE.PayloadError, "missing selection"):
            self.certify(evidence, expectation, logs)
        logs["worker"] = [event(1), event(1, payload="FullState")]
        with self.assertRaisesRegex(MODULE.PayloadError, "contradictory selections"):
            self.certify(evidence, expectation, logs)
        logs["worker"] = [event(1), event(1)]
        with self.assertRaisesRegex(MODULE.PayloadError, "duplicate selection"):
            self.certify(evidence, expectation, logs)

    def test_contradictory_exact_payload_kind_fails(self):
        evidence, expectation, logs, _ = fixture()
        expectation["stages"][1]["exact_payload_kind"] = "kv-recurrent"
        for kind in ("kv-recurrent", "kv-dense"):
            logs["worker"].append({
                "event": "stage.binary_kv_lookup_decision",
                "attributes": {
                    "skippy.run_id": "current-run",
                    "skippy.model_id": "pinned-model",
                    "skippy.topology_id": "topology",
                    "skippy.stage_id": "stage-1",
                    "skippy.stage_index": 1,
                    "skippy.exact_cache.payload_kind": kind,
                },
            })
        with self.assertRaisesRegex(MODULE.PayloadError, "exact payload"):
            self.certify(evidence, expectation, logs)

    def test_cli_records_warm_evidence_and_failure_artifact(self):
        evidence, expectation, logs, event = fixture()
        expectation["revision"] = "pinned-revision"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {
                "evidence.json": evidence,
                "expectations.json": {"models": {"pinned": expectation}},
                "manifest.json": {"artifacts": [{
                    "id": "pinned", "revision": "pinned-revision", "sha256": "a" * 64,
                }]},
                "roster.json": {"native_recipe": {"skippy_abi": "0.1.67"}},
            }
            for name, value in inputs.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            responses = root / "responses"
            responses.mkdir()
            runtime_dir = root / "native-runtimes" / "cpu"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "manifest.json").write_text(json.dumps({
                "runtime": {"id": "cpu", "skippy_abi": "0.1.67"}
            }), encoding="utf-8")
            for index, cached in ((1, 0), (2, 8)):
                (responses / f"response-{index}.json").write_text(json.dumps({
                    "id": f"request-{index}",
                    "choices": [{"message": {"content": "same continuation"}}],
                    "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": cached}},
                }), encoding="utf-8")
            command = [
                sys.executable, str(SCRIPT),
                "--evidence", str(root / "evidence.json"),
                "--expectations", str(root / "expectations.json"),
                "--model-manifest", str(root / "manifest.json"),
                "--roster", str(root / "roster.json"),
                "--runtime-bundle", str(root / "native-runtimes"),
                "--tested-commit", "head",
                "--artifact-id", "pinned",
                "--model-sha256", "a" * 64,
                "--seed-log", str(root / "seed.log"),
                "--worker-log", str(root / "worker.log"),
                "--responses-dir", str(responses),
                "--output", str(root / "result.json"),
            ]
            def run_with_worker(worker):
                (root / "seed.log").write_text(json.dumps(logs["seed"][0]) + "\n")
                (root / "worker.log").write_text(json.dumps(worker) + "\n")
                return subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(run_with_worker(logs["worker"][0]).returncode, 0)
            passed = json.loads((root / "result.json").read_text())
            self.assertEqual(passed["status"], "pass")
            self.assertEqual(passed["cold_warm"][1]["cached_tokens"], 8)
            self.assertNotEqual(run_with_worker(event(1, payload="FullState")).returncode, 0)
            failed = json.loads((root / "result.json").read_text())
            self.assertEqual(failed["status"], "fail")
            self.assertIn("stage stage-1 payload", failed["error"])


if __name__ == "__main__":
    unittest.main()
