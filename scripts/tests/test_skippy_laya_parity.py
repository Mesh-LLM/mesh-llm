"""The Laya parity comparison, checked against the vendored goldens without a model."""

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("laya_parity", ROOT / "scripts" / "skippy-laya-parity.py")
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)

FIXTURES = ROOT / "ci" / "llama-canary" / "fixtures" / "laya-golden"


def golden(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


class LayaParityTest(unittest.TestCase):
    def test_every_vendored_fixture_has_an_upstream_error_budget(self):
        names = {path.stem for path in FIXTURES.glob("*.json") if path.stem != "manifest"}
        self.assertEqual(names, set(parity.UPSTREAM_CPU_ERROR))

    def test_the_golden_answers_pass_against_themselves(self):
        for name in parity.UPSTREAM_CPU_ERROR:
            fixture = golden(name)
            ids = {key: q["input_ids"] for key, q in fixture["per_question"].items()}
            result = parity.compare(name, fixture, fixture["answers"], ids)
            self.assertEqual(result["failures"], [], name)

    def test_noul_error_within_the_upstream_budget_passes(self):
        fixture = golden("noul_zh")
        answers = json.loads(json.dumps(fixture["answers"]))
        (key,) = answers
        answers[key]["noul"] -= 0.058  # what upstream CPU and this PR both measure
        self.assertEqual(parity.compare("noul_zh", fixture, answers)["failures"], [])

    def test_a_larger_error_fails(self):
        fixture = golden("choice_multi_zh")
        answers = json.loads(json.dumps(fixture["answers"]))
        (key,) = answers
        option = next(iter(answers[key]["probabilities"]))
        answers[key]["probabilities"][option] += 0.05
        failures = parity.compare("choice_multi_zh", fixture, answers)["failures"]
        self.assertTrue(any("exceeds" in failure for failure in failures), failures)

    def test_a_changed_choice_or_token_ids_fail(self):
        fixture = golden("choice_single_en")
        answers = json.loads(json.dumps(fixture["answers"]))
        (key,) = answers
        other = next(name for name in answers[key]["probabilities"] if name != answers[key]["choice"])
        answers[key]["choice"] = other
        ids = {k: q["input_ids"][:-1] for k, q in fixture["per_question"].items()}
        failures = parity.compare("choice_single_en", fixture, answers, ids)
        joined = " ".join(failures["failures"])
        self.assertIn("choice", joined)
        self.assertIn("token ids", joined)


if __name__ == "__main__":
    unittest.main()
