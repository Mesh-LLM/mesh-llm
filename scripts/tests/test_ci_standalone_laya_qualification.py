from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "skippy/scripts/ci-standalone-laya-qualification.py"
SPEC = importlib.util.spec_from_file_location("ci_standalone_laya_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)


class DecisionsEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = "laya-test"
        self.system = {
            "model": self.model,
            "answers": {
                "question_0": {"noul": 0.8},
                "question_1": {"choice": "billing", "probabilities": {"billing": 0.7, "support": 0.3}},
                "question_2": {"score": 0.6, "probabilities": {"0": 0.4, "1": 0.6}},
            },
        }
        self.decisions = {
            "model": self.model,
            "answers": [
                {"type": "predicate", "name": "urgent", "probability": 0.8},
                {"type": "choice", "name": "team", "choice": "billing", "probabilities": [
                    {"value": "billing", "probability": 0.7},
                    {"value": "support", "probability": 0.3},
                ]},
                {"type": "score", "name": "frustration", "score": 0.6, "probabilities": [
                    {"value": 0, "label": "Calm", "probability": 0.4},
                    {"value": 1, "label": "Frustrated", "probability": 0.6},
                ]},
            ],
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }

    def test_equivalent_normalized_read_passes(self) -> None:
        qualifier.equivalence(self.system, self.decisions, self.model)

    def test_rejects_probability_and_label_corruption(self) -> None:
        for path, value in (
            ((0, "probability"), float("nan")),
            ((1, "choice"), "support"),
            ((1, "probabilities", 1, "probability"), 0.8),
            ((2, "probabilities", 1, "label"), "Other"),
            ((2, "score"), 0.9),
        ):
            with self.subTest(path=path):
                result = copy.deepcopy(self.decisions)
                target = result["answers"][path[0]]
                for key in path[1:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    qualifier.equivalence(self.system, result, self.model)


if __name__ == "__main__":
    unittest.main()
