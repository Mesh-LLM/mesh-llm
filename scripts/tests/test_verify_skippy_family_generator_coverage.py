from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


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
        manifest = self.write("manifest.json", {"models": [{"family": "alpha"}]})
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
        manifest = self.write("manifest.json", {"models": [{"family": "alpha"}]})
        family_map = self.write("map.json", {"families": {"alpha": ["src/models/alpha.cpp"]}})
        report = self.write("report.json", {"builders": [{
            "file": "src/models/alpha.cpp",
            "verdict": "transformable",
            "proof": {"execution_scope": "partitioned_decoder"},
        }]})
        self.assertEqual([], self.module.verify(manifest, family_map, report))


if __name__ == "__main__":
    unittest.main()
