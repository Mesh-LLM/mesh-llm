import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize-depot-registry-pulls.py"
SPEC = importlib.util.spec_from_file_location("registry_pulls", SCRIPT)
assert SPEC and SPEC.loader
REGISTRY_PULLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGISTRY_PULLS)


def observations(upstream: list[int], depot: list[int]) -> list[dict[str, object]]:
    digest = "sha256:" + "a" * 64
    return [
        {"source": source, "sample": index, "elapsed_ms": elapsed, "digest": digest}
        for source, values in (("upstream", upstream), ("depot", depot))
        for index, elapsed in enumerate(values, start=1)
    ]


class DepotRegistryPullSummaryTests(unittest.TestCase):
    def test_accepts_both_absolute_and_relative_improvement(self) -> None:
        result = REGISTRY_PULLS.summarize(
            observations([60_000] * 5, [40_000] * 5), 5
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["improvement_ms"], 20_000)

    def test_rejects_fast_percentage_only_improvement(self) -> None:
        result = REGISTRY_PULLS.summarize(
            observations([20_000] * 5, [14_000] * 5), 5
        )
        self.assertFalse(result["eligible"])

    def test_rejects_digest_mismatch(self) -> None:
        values = observations([60_000] * 5, [40_000] * 5)
        values[-1]["digest"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "different digests"):
            REGISTRY_PULLS.summarize(values, 5)

    def test_requires_unique_samples_per_source(self) -> None:
        values = observations([60_000] * 5, [40_000] * 5)
        values[-1]["sample"] = 4
        with self.assertRaisesRegex(ValueError, "unique samples"):
            REGISTRY_PULLS.summarize(values, 5)


if __name__ == "__main__":
    unittest.main()
