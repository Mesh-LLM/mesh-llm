"""Regression coverage for shard-scoped canary certification ports."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "canary_family_ports", ROOT / "scripts/lib/canary_family_ports.py"
)
PORTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PORTS)


class CanaryFamilyPortTests(unittest.TestCase):
    def manifest(self, rows: list[tuple[str, str]]) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "resolved-models.tsv"
        path.write_text(
            "family|class\n"
            + "".join(f"{family}|{model_class}\n" for family, model_class in rows),
            encoding="utf-8",
        )
        return path

    def test_single_causal_shard_ignores_unrelated_full_battery_ports(self) -> None:
        ports = PORTS.required_ports(
            self.manifest([("qwen2-vl", "causal_generation")]), 19337, 19338
        )
        self.assertEqual([19001, 19011, 19012, 19031, 19032], ports)
        for unrelated in (
            19547,
            19548,
            19557,
            19558,
            19567,
            19577,
            19578,
            19587,
            19588,
        ):
            self.assertNotIn(unrelated, ports)

    def test_model_order_and_workload_oracle_ports_match_execution(self) -> None:
        ports = PORTS.required_ports(
            self.manifest(
                [
                    ("embedding", "embedding"),
                    ("causal", "causal_generation"),
                    ("tts", "speech_synthesis"),
                ]
            ),
            19337,
            19338,
        )
        self.assertEqual(
            [19051, 19061, 19062, 19081, 19082, 19337, 19338], ports
        )

    def test_invalid_or_conflicting_workload_ports_fail_closed(self) -> None:
        path = self.manifest([("embedding", "embedding")])
        for candidate, oracle in ((0, 19338), (19337, 70000), (19337, 19337)):
            with self.subTest(candidate=candidate, oracle=oracle), self.assertRaises(
                ValueError
            ):
                PORTS.required_ports(path, candidate, oracle)

    def test_invalid_manifest_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown class"):
            PORTS.required_ports(self.manifest([("bad", "unknown")]), 19337, 19338)


if __name__ == "__main__":
    unittest.main()
