from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate-llama-canary-agent-manifests.py"


def load_module():
    spec = importlib.util.spec_from_file_location("llama_canary_manifest_policy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LlamaCanaryAgentManifestPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_module()
        self.family = {
            "schema_version": 1,
            "profiles": {"full": {"required_lanes": ["single-step"]}},
            "models": [
                {
                    "family": "alpha",
                    "cadences": ["llama-bump"],
                    "execution": {"trunk_layers": 2},
                    "resources": {
                        "runner_role": "family-certify",
                        "estimated_model_bytes": 100,
                    },
                }
            ],
        }
        self.parity = {
            "schema_version": 1,
            "goal": "classify every source",
            "candidates": [
                {"llama_model": "alpha", "family": "alpha", "status": "certified"}
            ],
        }

    def test_family_allows_only_positive_tensor_size_corrections(self) -> None:
        after = copy.deepcopy(self.family)
        after["models"][0]["resources"]["estimated_model_bytes"] = 96
        self.policy.validate_family_manifest(self.family, after)

        after["models"][0]["execution"]["trunk_layers"] = 1
        with self.assertRaisesRegex(self.policy.PolicyError, "changed outside"):
            self.policy.validate_family_manifest(self.family, after)

    def test_family_rejects_roster_changes(self) -> None:
        after = copy.deepcopy(self.family)
        after["models"].append(copy.deepcopy(after["models"][0]))
        with self.assertRaisesRegex(self.policy.PolicyError, "roster changed"):
            self.policy.validate_family_manifest(self.family, after)

    def test_parity_allows_exact_additive_classification(self) -> None:
        after = copy.deepcopy(self.parity)
        after["candidates"].append(
            {"llama_model": "beta", "family": "beta", "status": "candidate"}
        )
        self.policy.validate_parity_manifest(
            self.parity,
            after,
            {"alpha", "beta"},
            {"alpha", "beta"},
        )

    def test_parity_rejects_existing_row_changes_and_missing_sources(self) -> None:
        changed = copy.deepcopy(self.parity)
        changed["candidates"][0]["status"] = "needs_candidate"
        with self.assertRaisesRegex(self.policy.PolicyError, "existing parity"):
            self.policy.validate_parity_manifest(
                self.parity,
                changed,
                {"alpha"},
                {"alpha"},
            )

        with self.assertRaisesRegex(self.policy.PolicyError, "exactly classify"):
            self.policy.validate_parity_manifest(
                self.parity,
                copy.deepcopy(self.parity),
                {"alpha", "beta"},
                {"alpha", "beta"},
            )

    def test_parity_rejects_scope_reduction_and_new_artifact_authority(self) -> None:
        after = copy.deepcopy(self.parity)
        after["candidates"].append(
            {
                "llama_model": "beta",
                "family": "beta",
                "status": "needs_boundary_registration",
            }
        )
        with self.assertRaisesRegex(self.policy.PolicyError, "runnable candidate"):
            self.policy.validate_parity_manifest(
                self.parity,
                after,
                {"alpha", "beta"},
                {"alpha", "beta"},
            )

        after["candidates"][-1] = {
            "llama_model": "beta",
            "family": "beta",
            "status": "candidate",
            "model_pin": {"repo": "owner/repo"},
        }
        with self.assertRaisesRegex(self.policy.PolicyError, "classification metadata only"):
            self.policy.validate_parity_manifest(
                self.parity,
                after,
                {"alpha", "beta"},
                {"alpha", "beta"},
            )

    def test_parity_rejects_non_classification_fields(self) -> None:
        for field, value in (
            ("repo", "owner/repo"),
            ("include", "*.gguf"),
            ("revision", "a" * 40),
            ("file_integrity", {"model.gguf": {"size_bytes": 1}}),
            ("splits", "1"),
            ("recurrent", "all"),
        ):
            with self.subTest(field=field):
                after = copy.deepcopy(self.parity)
                after["candidates"].append(
                    {
                        "llama_model": "beta",
                        "family": "beta",
                        "status": "needs_boundary_registration",
                        field: value,
                    }
                )
                with self.assertRaisesRegex(self.policy.PolicyError, "classification metadata only"):
                    self.policy.validate_parity_manifest(
                        self.parity,
                        after,
                        {"alpha", "beta"},
                        {"alpha", "beta"},
                    )


if __name__ == "__main__":
    unittest.main()
