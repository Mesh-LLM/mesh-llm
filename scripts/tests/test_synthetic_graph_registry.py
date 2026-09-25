"""Keep executable sparse GGUF graph tests complete against the canary registry."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "third_party/llama.cpp/patches/0024-test-skippy-cover-the-complete-canary-graph-registry.patch"
CASE = re.compile(r"^\+skippy_contract_case\(([^)]+)\)$", re.MULTILINE)


def coverage_errors(models, cases):
    expected = {row["family"]: row for row in models}
    actual = {}
    errors = []
    for case in cases:
        family, arch, layers, width, mtp, mode = case.split()
        if family in actual:
            errors.append(f"duplicate fixture: {family}")
        actual[family] = (arch, int(layers), int(width), int(mtp), mode)
    for family in sorted(expected.keys() - actual.keys()):
        errors.append(f"missing fixture: {family}")
    for family in sorted(actual.keys() - expected.keys()):
        errors.append(f"unregistered fixture: {family}")
    for family in sorted(expected.keys() & actual.keys()):
        row = expected[family]
        execution = row["execution"]
        dimensions = (row["architecture"], execution["trunk_layers"],
                      execution["activation_width"], execution["mtp_layers"])
        if actual[family][:4] != dimensions:
            errors.append(f"registry dimensions differ: {family}")
        mode = "causal"
        if row["architecture"] == "gemma3n":
            mode = "shared-kv-18"
        elif row["architecture"] == "gemma4":
            mode = "shared-kv-22"
        elif row["architecture"] == "graniteswitch":
            mode = "causal-tokens"
        elif row["class"] in ("embedding", "rerank"):
            mode = "stateless"
        elif row["class"] == "encoder_decoder":
            mode = "encoder-decoder-rejection"
        if actual[family][4] != mode:
            errors.append(f"contract mode differs: {family}")
    return errors


class SyntheticGraphRegistryTests(unittest.TestCase):
    def setUp(self):
        self.models = json.loads((ROOT / "ci/llama-canary/family-certified.json").read_text())["models"]
        self.cases = CASE.findall(PATCH.read_text())

    def test_every_registry_family_has_an_executable_fixture(self):
        self.assertEqual(coverage_errors(self.models, self.cases), [])

    def test_new_family_requires_fixture(self):
        added = dict(self.models[0], family="new-canary-family")
        self.assertIn("missing fixture: new-canary-family", coverage_errors(self.models + [added], self.cases))

    def test_removed_fixture_fails(self):
        self.assertTrue(coverage_errors(self.models, self.cases[1:]))

    def test_duplicate_fixture_fails(self):
        self.assertTrue(coverage_errors(self.models, self.cases + self.cases[:1]))

    def test_dimension_and_mtp_drift_fail(self):
        for field in ("trunk_layers", "activation_width", "mtp_layers"):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(self.models))
                changed[0]["execution"][field] += 1
                self.assertIn("registry dimensions differ: " + changed[0]["family"], coverage_errors(changed, self.cases))

    def test_cases_are_consumed_by_ctest(self):
        patch = PATCH.read_text()
        self.assertIn('+    include(skippy/tests/graph_contract_cases.cmake)', patch)
        self.assertIn('+    add_test(NAME skippy_graph_contract_${family} COMMAND skippy-graph-contract-models', patch)
        self.assertIn('FIXTURES_REQUIRED contract-${family}', patch)
        self.assertNotIn('DISABLED TRUE', patch)
        self.assertNotIn('WILL_FAIL', patch)


if __name__ == "__main__":
    unittest.main()
