from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "skippy/scripts/validate-ci-qualification.py"
SPEC = importlib.util.spec_from_file_location("validate_ci_qualification", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)
ASSEMBLER_PATH = ROOT / "skippy/scripts/assemble-ci-qualification.py"
ASSEMBLER_SPEC = importlib.util.spec_from_file_location("assemble_ci_qualification", ASSEMBLER_PATH)
assert ASSEMBLER_SPEC is not None and ASSEMBLER_SPEC.loader is not None
assembler = importlib.util.module_from_spec(ASSEMBLER_SPEC)
ASSEMBLER_SPEC.loader.exec_module(assembler)
SHA = "a" * 40
PLAN = "b" * 64
REGISTRY = json.loads((ROOT / "ci/model-artifacts/registry.json").read_text(encoding="utf-8"))
MODEL_HASHES = {
    item["id"]: item["artifact"]["files"][0]["sha256"]
    for item in REGISTRY["artifacts"]
    if len(item["artifact"].get("files", [])) == 1
}
SUITE_MODELS = {
    "packaging-runtime": {},
    "dense": {"family-qwen3-dense": MODEL_HASHES["family-qwen3-dense"]},
    "recurrent": {"family-granite-hybrid": MODEL_HASHES["family-granite-hybrid"]},
    "moe": {"family-granite-moe": MODEL_HASHES["family-granite-moe"]},
    "kv-cache": {
        "family-qwen3-dense": MODEL_HASHES["family-qwen3-dense"],
        "family-granite-hybrid": MODEL_HASHES["family-granite-hybrid"],
    },
    "system-one": {
        "family-laya-multilingual": MODEL_HASHES["family-laya-multilingual"],
    },
    "decisions": {
        "family-laya-multilingual": MODEL_HASHES["family-laya-multilingual"],
    },
}


class CiQualificationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.product_path = base / "product-manifest.json"
        self.product_dir = base / "product"
        runtime_dir = self.product_dir / "native-runtimes" / "runtime-a"
        runtime_dir.mkdir(parents=True)
        (self.product_dir / "skippy").write_bytes(b"test standalone CLI")
        (self.product_dir / "build-contract.json").write_bytes(b"test CLI contract")
        (self.product_dir / "host-imports.json").write_bytes(b"test import report")
        (runtime_dir / "manifest.json").write_bytes(b"test runtime manifest")
        (runtime_dir / "libllama.so").write_bytes(b"test native runtime")
        self.availability_path = base / "availability.json"
        self.evidence_dir = base / "evidence"
        self.evidence_dir.mkdir()
        self.product = {
            "schema_version": 1,
            "contract": "skippy-product-v1",
            "source_sha": SHA,
            "target": "x86_64-unknown-linux-gnu",
            "backend": "cuda",
            "cli": {
                "path": "skippy",
                "sha256": contract.digest(self.product_dir / "skippy"),
                "build_contract_sha256": contract.digest(self.product_dir / "build-contract.json"),
                "host_imports_sha256": contract.digest(self.product_dir / "host-imports.json"),
            },
            "runtime": {
                "path": "native-runtimes/runtime-a",
                "manifest_sha256": contract.digest(runtime_dir / "manifest.json"),
                "sha256": contract.tree_digest(runtime_dir),
            },
        }
        self.availability = {
            "schema_version": 1,
            "source_sha": SHA,
            "plan_digest": PLAN,
            "row_id": "linux-cuda",
            "policy_source": "protected-ci",
            "state": "available",
            "runner": "gpu-nvidia",
        }
        self.receipt = {
            "schema_version": 1,
            "source_sha": SHA,
            "plan_digest": PLAN,
            "row_id": "linux-cuda",
            "status": "qualified",
            "hardware": {
                "actual_backend": "cuda", "runner": "gpu-nvidia", "device": "NVIDIA GPU",
                "driver": "test-driver", "runtime_version": "test-runtime",
                "device_architecture": "sm_86", "offloaded_layers": 16,
            },
            "suites": {
                name: {
                    "status": "passed", "cases": sorted(cases),
                    "evidence_sha256": "", "models": copy.deepcopy(SUITE_MODELS[name]),
                }
                for name, cases in contract.SUITE_CASES.items()
            },
        }
        for name, result in self.receipt["suites"].items():
            result["evidence_file"] = f"{name}.json"
        self.refresh_hashes()

    def refresh_hashes(self) -> None:
        self.product_path.write_text(json.dumps(self.product), encoding="utf-8")
        (self.product_dir / "product-manifest.json").write_bytes(self.product_path.read_bytes())
        self.availability_path.write_text(json.dumps(self.availability), encoding="utf-8")
        self.receipt["product_manifest_sha256"] = hashlib.sha256(self.product_path.read_bytes()).hexdigest()
        self.receipt["availability_sha256"] = hashlib.sha256(self.availability_path.read_bytes()).hexdigest()
        for name, result in self.receipt["suites"].items():
            if result["status"] == "passed":
                path = self.evidence_dir / f"{name}.json"
                evidence = {
                    "schema_version": 1, "status": "passed", "source_sha": SHA,
                    "row_id": self.availability["row_id"],
                    "suite": name, "product_manifest_sha256": self.receipt["product_manifest_sha256"],
                    "executed_cases": result["cases"],
                }
                evidence["models"] = result["models"]
                path.write_text(json.dumps(evidence), encoding="utf-8")
                result["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def validate(self) -> None:
        contract.validate_receipt(
            self.receipt, self.product, self.product_path, self.product_dir, self.availability,
            self.availability_path, self.evidence_dir, source_sha=SHA, plan_digest=PLAN,
            row_id=self.availability["row_id"],
        )

    def assemble(self) -> dict:
        return assembler.assemble(
            product_dir=self.product_dir, availability_path=self.availability_path,
            evidence_dir=self.evidence_dir,
            hardware_path=self.hardware_path,
            source_sha=SHA, plan_digest=PLAN, row_id=self.availability["row_id"],
        )

    def test_assembler_accepts_exact_executed_suite_set(self) -> None:
        self.hardware_path = Path(self.temporary.name) / "hardware.json"
        self.hardware_path.write_text(json.dumps(self.receipt["hardware"]), encoding="utf-8")
        self.assertEqual(self.assemble(), self.receipt)

    def test_assembler_rejects_missing_or_failed_suite_evidence(self) -> None:
        self.hardware_path = Path(self.temporary.name) / "hardware.json"
        self.hardware_path.write_text(json.dumps(self.receipt["hardware"]), encoding="utf-8")
        path = self.evidence_dir / "moe.json"
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaises(OSError):
            self.assemble()
        path.write_bytes(saved)
        evidence = json.loads(saved)
        evidence["status"] = "failed"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "moe evidence did not pass"):
            self.assemble()

    def test_assembler_requires_packaging_evidence_on_unavailable_row(self) -> None:
        self.product["backend"] = "rocm"
        self.availability["row_id"] = "linux-rocm"
        self.availability.update({"state": "hardware-unavailable", "reason": "no approved GPU runner"})
        self.availability.pop("runner")
        self.refresh_hashes()
        self.hardware_path = None
        receipt = self.assemble()
        self.assertEqual(receipt["status"], "hardware-unavailable")
        self.assertEqual(receipt["suites"]["dense"], {"status": "not-executed"})
        (self.evidence_dir / "packaging-runtime.json").unlink()
        with self.assertRaises(OSError):
            self.assemble()

    def test_all_nine_core_rows_match_runtime_catalog(self) -> None:
        for row_id in contract.CORE_ROWS:
            with self.subTest(row_id=row_id):
                self.assertEqual(contract.catalog_row(row_id)["id"], row_id)

    def test_every_audit_case_is_required_or_explicitly_deferred(self) -> None:
        audit_cases = {
            "packaging-runtime": {"archive-integrity", "imports", "abi", "discovery", "version", "no-driver"},
            "dense": {"load", "prefill-decode", "stream", "stop-cancel", "concurrent", "continuation", "restart", "staged-parity"},
            "recurrent": {"prefill-decode", "state-preservation", "repeated-restore", "reset-isolation", "restart"},
            "moe": {"expert-metadata", "expert-execution", "staged-parity", "repeated-restore", "suffix-continuation", "session-isolation"},
            "kv-cache": {"dense-prefix-hit", "recurrent-prefix-hit", "suffix-continuation", "divergent-prefix", "isolation", "eviction", "import-export", "persisted-restart", "corrupt-rejection", "restore-observed"},
            "system-one": {"laya-goldens", "reader-contract", "negative-cases", "lifecycle"},
            "decisions": {"endpoint-equivalence", "probability-contract", "negative-cases", "lifecycle"},
        }
        self.assertEqual(set(contract.SUITE_CASES), set(contract.DEFERRED_CASES))
        for suite, cases in audit_cases.items():
            with self.subTest(suite=suite):
                required = contract.SUITE_CASES[suite]
                deferred = set(contract.DEFERRED_CASES[suite])
                self.assertEqual(required | deferred, cases,
                                 f"{suite} silently dropped or invented an audit case")
                self.assertFalse(required & deferred, f"{suite} case is both required and deferred")
                self.assertTrue(all(reason.strip() for reason in contract.DEFERRED_CASES[suite].values()),
                                f"{suite} deferred case lacks a reason")

    def test_system_one_and_decisions_have_separate_required_results(self) -> None:
        self.assertIn("system-one", contract.REQUIRED_SUITES)
        self.assertIn("decisions", contract.REQUIRED_SUITES)
        self.assertNotIn("system-one-decisions", contract.REQUIRED_SUITES)
        self.receipt["suites"].pop("decisions")
        with self.assertRaisesRegex(ValueError, "missing, duplicate, or unknown qualification suite"):
            self.validate()

    def test_qualified_gpu_receipt_requires_all_suites_and_actual_device(self) -> None:
        self.validate()
        for change in ("missing-suite", "missing-case", "wrong-model", "missing-kv-model", "cpu-fallback", "wrong-runner", "zero-offload", "software-renderer", "wrong-plan", "wrong-product", "tampered-evidence"):
            with self.subTest(change=change):
                saved = copy.deepcopy(self.receipt)
                if change == "missing-suite":
                    self.receipt["suites"].pop("moe")
                elif change == "missing-case":
                    self.receipt["suites"]["moe"]["cases"].remove("expert-execution")
                elif change == "wrong-model":
                    self.receipt["suites"]["moe"]["models"]["family-granite-moe"] = "e" * 64
                elif change == "missing-kv-model":
                    self.receipt["suites"]["kv-cache"]["models"].pop("family-granite-hybrid")
                elif change == "cpu-fallback":
                    self.receipt["hardware"]["actual_backend"] = "cpu"
                elif change == "wrong-runner":
                    self.receipt["hardware"]["runner"] = "unapproved-gpu"
                elif change == "zero-offload":
                    self.receipt["hardware"]["offloaded_layers"] = 0
                elif change == "software-renderer":
                    self.receipt["hardware"]["device"] = "lavapipe"
                elif change == "wrong-plan":
                    self.receipt["plan_digest"] = "e" * 64
                elif change == "wrong-product":
                    self.receipt["product_manifest_sha256"] = "e" * 64
                else:
                    self.receipt["suites"]["dense"]["evidence_sha256"] = "e" * 64
                with self.assertRaises(ValueError):
                    self.validate()
                self.receipt = saved

    def test_unavailable_hardware_cannot_report_positive_execution(self) -> None:
        self.product["backend"] = "rocm"
        self.availability["row_id"] = "linux-rocm"
        self.receipt["row_id"] = "linux-rocm"
        self.availability.update({"state": "hardware-unavailable", "reason": "no approved GPU runner"})
        self.availability.pop("runner")
        self.receipt["status"] = "hardware-unavailable"
        self.receipt["hardware"] = None
        packaging = self.receipt["suites"]["packaging-runtime"]
        self.receipt["suites"] = {name: {"status": "not-executed"} for name in contract.REQUIRED_SUITES}
        self.receipt["suites"]["packaging-runtime"] = packaging
        self.refresh_hashes()
        self.validate()
        self.receipt["suites"]["dense"] = {"status": "passed", "cases": ["fake"]}
        with self.assertRaises(ValueError):
            self.validate()

    def test_unavailable_hardware_still_requires_packaging_checks(self) -> None:
        self.product["backend"] = "rocm"
        self.availability["row_id"] = "linux-rocm"
        self.receipt["row_id"] = "linux-rocm"
        self.availability.update({"state": "hardware-unavailable", "reason": "no approved GPU runner"})
        self.availability.pop("runner")
        self.receipt["status"] = "hardware-unavailable"
        self.receipt["hardware"] = None
        self.receipt["suites"] = {name: {"status": "not-executed"} for name in contract.REQUIRED_SUITES}
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "packaging-runtime did not pass"):
            self.validate()

    def test_unavailable_cpu_is_rejected(self) -> None:
        self.product["backend"] = "cpu"
        self.availability.update({"row_id": "linux-cpu", "state": "hardware-unavailable", "reason": "disabled"})
        self.receipt.update({"row_id": "linux-cpu", "status": "hardware-unavailable"})
        self.receipt["hardware"] = None
        self.receipt["suites"] = {name: {"status": "not-executed"} for name in contract.REQUIRED_SUITES}
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "required available row cannot be hardware-unavailable"):
            contract.validate_receipt(
                self.receipt, self.product, self.product_path, self.product_dir, self.availability,
                self.availability_path, self.evidence_dir, source_sha=SHA, plan_digest=PLAN,
                row_id="linux-cpu",
            )

    def test_policy_available_accelerator_row_cannot_claim_unavailable(self) -> None:
        self.availability.update({"state": "available", "runner": "gpu-nvidia"})
        self.receipt["status"] = "hardware-unavailable"
        self.receipt["hardware"] = None
        self.receipt["suites"] = {name: {"status": "not-executed"} for name in contract.REQUIRED_SUITES}
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "available row must be qualified"):
            self.validate()

    def test_accelerator_row_may_be_recorded_unavailable_by_policy(self) -> None:
        self.availability.update({"state": "hardware-unavailable",
                                  "reason": "MESH_CUDA_INFERENCE_RUNNER_ENABLED is not exactly true"})
        self.availability.pop("runner")
        self.receipt["status"] = "hardware-unavailable"
        self.receipt["hardware"] = None
        packaging = self.receipt["suites"]["packaging-runtime"]
        self.receipt["suites"] = {name: {"status": "not-executed"} for name in contract.REQUIRED_SUITES}
        self.receipt["suites"]["packaging-runtime"] = packaging
        self.refresh_hashes()
        self.validate()
        self.assertEqual(self.receipt["status"], "hardware-unavailable")

    def test_metal_row_cannot_be_downgraded_to_unavailable(self) -> None:
        self.product.update({"backend": "metal", "target": "aarch64-apple-darwin"})
        self.availability.update({"row_id": "macos-metal", "state": "hardware-unavailable", "reason": "runner disappeared"})
        self.availability.pop("runner")
        self.receipt.update({"row_id": "macos-metal", "status": "hardware-unavailable", "hardware": None})
        self.receipt["suites"] = {name: {"status": "not-executed"} for name in contract.REQUIRED_SUITES}
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "required available row cannot be hardware-unavailable"):
            contract.validate_receipt(
                self.receipt, self.product, self.product_path, self.product_dir, self.availability,
                self.availability_path, self.evidence_dir, source_sha=SHA, plan_digest=PLAN,
                row_id="macos-metal",
            )

    def test_duplicate_json_key_cannot_hide_a_failed_suite(self) -> None:
        path = Path(self.temporary.name) / "duplicate.json"
        path.write_text('{"status":"failed","status":"passed"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            contract.load_json(path)

    def test_failed_evidence_cannot_be_labeled_as_passed_in_receipt(self) -> None:
        path = self.evidence_dir / "dense.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["status"] = "failed"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        self.receipt["suites"]["dense"]["evidence_sha256"] = contract.digest(path)
        with self.assertRaisesRegex(ValueError, "dense evidence did not pass"):
            self.validate()

    def test_changed_product_bytes_fail_even_when_manifest_is_unchanged(self) -> None:
        (self.product_dir / "skippy").write_bytes(b"corrupted CLI")
        with self.assertRaisesRegex(ValueError, "standalone CLI bytes differ"):
            self.validate()

    def test_changed_import_report_fails_even_when_manifest_is_unchanged(self) -> None:
        (self.product_dir / "host-imports.json").write_bytes(b"corrupted import report")
        with self.assertRaisesRegex(ValueError, "import report bytes differ"):
            self.validate()

    def test_changed_runtime_bytes_fail_even_when_manifest_is_unchanged(self) -> None:
        (self.product_dir / "native-runtimes/runtime-a/libllama.so").write_bytes(b"corrupted runtime")
        with self.assertRaisesRegex(ValueError, "standalone runtime bytes differ"):
            self.validate()

    def test_escaping_product_path_is_rejected(self) -> None:
        self.product["cli"]["path"] = "../skippy"
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "path escapes bundle"):
            self.validate()

    def test_symlinked_product_parent_is_rejected(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (outside / "skippy").write_bytes((self.product_dir / "skippy").read_bytes())
        (self.product_dir / "linked").symlink_to(outside, target_is_directory=True)
        self.product["cli"]["path"] = "linked/skippy"
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "path contains a symlink"):
            self.validate()

    def test_symlinked_runtime_parent_is_rejected(self) -> None:
        runtime_root = self.product_dir / "native-runtimes"
        runtime_root.rename(self.product_dir / "moved-runtimes")
        runtime_root.symlink_to(self.product_dir / "moved-runtimes", target_is_directory=True)
        self.refresh_hashes()
        with self.assertRaisesRegex(ValueError, "runtime path contains a symlink"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
