"""Contract checks for the Mesh host standalone-receipt consumer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci-host-qualification-receipts.py"
SPEC = importlib.util.spec_from_file_location("ci_host_qualification_receipts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consumer)

SHA = "a" * 40
PLAN = "b" * 64


def receipt(row_id: str, *, status: str = "qualified", source: str = SHA, plan: str = PLAN) -> dict:
    return {"schema_version": 1, "source_sha": source, "plan_digest": plan,
            "row_id": row_id, "status": status}


class HostReceiptConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.receipts = Path(self.temporary.name)
        self.output = self.receipts / "record.json"

    def write(self, name: str, value: dict) -> None:
        (self.receipts / name).write_text(json.dumps(value), encoding="utf-8")

    def verify(self, platform: str = "linux") -> dict:
        return consumer.verify(self.receipts, source_sha=SHA, plan_digest=PLAN, platform=platform)

    def test_accepts_a_qualified_required_row(self) -> None:
        self.write("linux-cpu.json", receipt("linux-cpu"))
        record = self.verify()
        self.assertEqual(record["required_row"], "linux-cpu")
        self.assertEqual([entry["row_id"] for entry in record["receipts"]], ["linux-cpu"])
        self.assertEqual(len(record["receipts"][0]["receipt_sha256"]), 64)

    def test_accepts_an_explicit_unavailable_accelerator_row_alongside(self) -> None:
        self.write("linux-cpu.json", receipt("linux-cpu"))
        self.write("linux-cuda.json", receipt("linux-cuda", status="hardware-unavailable"))
        record = self.verify()
        self.assertEqual({entry["status"] for entry in record["receipts"]},
                         {"qualified", "hardware-unavailable"})

    def test_rejects_an_empty_receipt_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "no standalone qualification receipts"):
            self.verify()

    def test_rejects_a_foreign_source_revision(self) -> None:
        self.write("linux-cpu.json", receipt("linux-cpu", source="c" * 40))
        with self.assertRaisesRegex(ValueError, "another source revision"):
            self.verify()

    def test_rejects_a_foreign_plan(self) -> None:
        self.write("linux-cpu.json", receipt("linux-cpu", plan="c" * 64))
        with self.assertRaisesRegex(ValueError, "another CI plan"):
            self.verify()

    def test_rejects_a_receipt_from_another_platform(self) -> None:
        self.write("windows-cpu.json", receipt("windows-cpu"))
        with self.assertRaisesRegex(ValueError, "another platform"):
            self.verify("linux")

    def test_rejects_an_unregistered_row(self) -> None:
        self.write("bogus.json", receipt("linux-bogus"))
        with self.assertRaisesRegex(ValueError, "unknown core row"):
            self.verify()

    def test_requires_the_platform_mandatory_row(self) -> None:
        self.write("linux-cuda.json", receipt("linux-cuda", status="hardware-unavailable"))
        with self.assertRaisesRegex(ValueError, "linux-cpu produced no receipt"):
            self.verify()

    def test_rejects_an_unqualified_mandatory_row(self) -> None:
        self.write("linux-cpu.json", receipt("linux-cpu", status="hardware-unavailable"))
        with self.assertRaisesRegex(ValueError, "linux-cpu is not qualified"):
            self.verify()

    def test_rejects_duplicate_receipts_for_one_row(self) -> None:
        self.write("a.json", receipt("linux-cpu"))
        self.write("b.json", receipt("linux-cpu"))
        with self.assertRaisesRegex(ValueError, "duplicate receipt"):
            self.verify()

    def test_rejects_an_unknown_status(self) -> None:
        self.write("linux-cpu.json", receipt("linux-cpu", status="pending"))
        with self.assertRaisesRegex(ValueError, "invalid qualification status"):
            self.verify()

    def test_rejects_an_unknown_platform(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown platform"):
            self.verify("bsd")


if __name__ == "__main__":
    unittest.main()
