"""A passing Cargo batch must account for each selected package exactly once."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "record-ci-cargo-census.py"
spec = importlib.util.spec_from_file_location("record_ci_cargo_census", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CargoCensusTests(unittest.TestCase):
    def test_accepts_actual_batch_in_any_execution_order(self):
        module.validate(["skippy-ffi", "mesh-llm"], ["mesh-llm", "skippy-ffi"])

    def test_rejects_missing_or_extra_package(self):
        with self.assertRaisesRegex(ValueError, "missing=\\['skippy-ffi'\\]"):
            module.validate(["mesh-llm", "skippy-ffi"], ["mesh-llm"])
        with self.assertRaisesRegex(ValueError, "unexpected=\\['skippy-ffi'\\]"):
            module.validate(["mesh-llm"], ["mesh-llm", "skippy-ffi"])

    def test_rejects_duplicate_execution(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.validate(["mesh-llm"], ["mesh-llm", "mesh-llm"])


if __name__ == "__main__":
    unittest.main()
