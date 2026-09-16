from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify-skippy-family-generator-coverage.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_generator_coverage", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratorCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_primary_family_requires_partitioned_decoder_transform(self) -> None:
        """Auxiliary sidecar support cannot substitute for a causal partitioned-decoder transform."""
        manifest = self.write("manifest.json", {"models": [{"family": "alpha", "class": "causal_generation", "profile": "full"}]})
        family_map = self.write("map.json", {"families": {"alpha": ["src/models/alpha.cpp"]}})
        report = self.write("report.json", {"builders": [{
            "file": "src/models/alpha.cpp",
            "verdict": "supported_auxiliary",
            "proof": {"execution_scope": "final_stage_sidecar"},
        }]})
        self.assertEqual(
            ["alpha: no mapped partitioned decoder was transformed"],
            self.module.verify(manifest, family_map, report),
        )

    def test_mapped_partitioned_decoder_passes(self) -> None:
        """Accept a family only when its mapped source has partitioned-decoder transformation proof."""
        manifest = self.write("manifest.json", {"models": [{"family": "alpha", "class": "causal_generation", "profile": "full"}]})
        family_map = self.write("map.json", {"families": {"alpha": ["src/models/alpha.cpp"]}})
        report = self.write("report.json", {"builders": [{
            "file": "src/models/alpha.cpp",
            "verdict": "transformable",
            "proof": {"execution_scope": "partitioned_decoder"},
        }]})
        self.assertEqual([], self.module.verify(manifest, family_map, report))

    def test_non_chat_rows_require_workload_profiles_not_decoder_transforms(self) -> None:
        """Full-model workload evidence cannot be confused with staged decoder proof."""
        family_map = self.write("map.json", {"families": {}})
        report = self.write("report.json", {"builders": []})
        for model_class in sorted(self.module.NON_CHAT_CLASSES):
            for profile in ("workload-smoke", "workload-oracle"):
                with self.subTest(model_class=model_class, profile=profile):
                    manifest = self.write("manifest.json", {"models": [{
                        "family": "non-chat", "class": model_class, "profile": profile,
                    }]})
                    self.assertEqual([], self.module.verify(manifest, family_map, report))

    def test_missing_unknown_and_misclassified_rows_fail_closed(self) -> None:
        """Changing a class or profile cannot silently bypass generator coverage."""
        family_map = self.write("map.json", {"families": {}})
        report = self.write("report.json", {"builders": []})
        for fields in ({}, {"class": "future"}, {"class": "embedding", "profile": "full"},
                       {"class": "causal_generation", "profile": "workload-oracle"}):
            with self.subTest(fields=fields):
                manifest = self.write("manifest.json", {"models": [{"family": "alpha", **fields}]})
                self.assertEqual(1, len(self.module.verify(manifest, family_map, report)))


if __name__ == "__main__":
    unittest.main()
