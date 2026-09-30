from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "verify-mesh-tdx-peer.py"
SPEC = importlib.util.spec_from_file_location("verify_mesh_tdx_peer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def fixture():
    nonce = bytes.fromhex("11" * 32)
    peer = bytes.fromhex("22" * 32)
    app_compose = '{"docker_compose_file":"pinned-image"}'
    compose_hash = hashlib.sha256(app_compose.encode()).digest()
    event = {
        "imr": 3,
        "event_type": 0x08000001,
        "event": "compose-hash",
        "event_payload": compose_hash.hex(),
        "digest": "",
    }
    digest = hashlib.sha384(
        event["event_type"].to_bytes(4, "little")
        + b":compose-hash:"
        + compose_hash
    ).digest()
    rtmr3 = hashlib.sha384(bytes(48) + digest).digest()
    policy = {
        "mr_td": bytes.fromhex("33" * 48),
        "rt_mr0": bytes.fromhex("44" * 48),
        "rt_mr1": bytes.fromhex("55" * 48),
        "rt_mr2": bytes.fromhex("66" * 48),
        "rt_mr3": rtmr3,
        "app_compose_sha256": compose_hash,
    }
    evidence = {
        "format": VERIFY.FORMAT,
        "nonce": nonce.hex(),
        "endpoint_id": peer.hex(),
        "quote": "aabbccdd",
        "app_compose": app_compose,
        "compose_hash": compose_hash.hex(),
        "event_log": [event],
    }
    report = SimpleNamespace(
        td_attributes=bytes(8),
        report_data=VERIFY.report_data(nonce, peer),
        **{name: policy[name] for name in VERIFY.MEASUREMENTS},
    )
    return evidence, nonce, peer, policy, report


class FakeQvl:
    def __init__(self, report):
        self.report = report
        self.status = "UpToDate"
        self.advisory_ids = []
        self.is_tdx = True
        self.valid_quote = bytes.fromhex("aabbccdd")
        self.verified = False
        self.Quote = SimpleNamespace(parse=self.parse)

    def verify(self, quote_bytes, _collateral, _now):
        if quote_bytes != self.valid_quote:
            raise ValueError("Intel signature verification failed")
        self.verified = True
        return SimpleNamespace(status=self.status, advisory_ids=self.advisory_ids)

    def parse(self, _quote_bytes):
        if not self.verified:
            raise AssertionError("parsed claims used before signature verification")
        return SimpleNamespace(report=self.report, is_tdx=lambda: self.is_tdx)


class VerifyMeshTdxPeerTests(unittest.TestCase):
    def setUp(self):
        self.evidence, self.nonce, self.peer, self.policy, report = fixture()
        self.qvl = FakeQvl(report)

    def verify(self):
        return VERIFY.verify_evidence(
            self.evidence, self.nonce, self.peer, self.policy,
            object(), self.qvl, 1_000_000,
        )

    def test_accepts_vendor_verified_quote_with_exact_peer_and_policy(self):
        result = self.verify()
        self.assertTrue(self.qvl.verified)
        self.assertEqual(result["endpoint_id"], self.peer.hex())
        self.assertEqual(result["status"], "verified")

    def test_rejects_wrong_challenge_before_quote_verification(self):
        self.evidence["nonce"] = (bytes.fromhex("99" * 32)).hex()
        with self.assertRaisesRegex(ValueError, "nonce differs"):
            self.verify()
        self.assertFalse(self.qvl.verified)

    def test_rejects_selected_peer_swap(self):
        self.evidence["endpoint_id"] = (bytes.fromhex("99" * 32)).hex()
        with self.assertRaisesRegex(ValueError, "selected Mesh peer"):
            self.verify()

    def test_rejects_quote_bound_to_different_endpoint(self):
        self.qvl.report.report_data = VERIFY.report_data(
            self.nonce, bytes.fromhex("99" * 32)
        )
        with self.assertRaisesRegex(ValueError, "does not bind"):
            self.verify()

    def test_rejects_quote_that_fails_intel_signature_check(self):
        self.evidence["quote"] = "aabbccde"
        with self.assertRaisesRegex(ValueError, "signature verification failed"):
            self.verify()

    def test_rejects_debug_and_old_tcb(self):
        self.qvl.report.td_attributes = b"\x01" + bytes(7)
        with self.assertRaisesRegex(ValueError, "debug mode"):
            self.verify()
        self.qvl.report.td_attributes = bytes(8)
        self.qvl.status = "OutOfDate"
        with self.assertRaisesRegex(ValueError, "TCB is not up to date"):
            self.verify()

    def test_rejects_advisories(self):
        self.qvl.advisory_ids = ["INTEL-SA-test"]
        with self.assertRaisesRegex(ValueError, "free of advisories"):
            self.verify()

    def test_rejects_unapproved_boot_measurement(self):
        self.qvl.report.mr_td = bytes.fromhex("aa" * 48)
        with self.assertRaisesRegex(ValueError, "mr_td differs"):
            self.verify()

    def test_rejects_unapproved_compose_and_tampered_event_log(self):
        self.evidence["app_compose"] = "other application"
        with self.assertRaisesRegex(ValueError, "application Compose differs"):
            self.verify()
        self.evidence["app_compose"] = fixture()[0]["app_compose"]
        self.evidence["event_log"] = copy.deepcopy(self.evidence["event_log"])
        self.evidence["event_log"][0]["event_payload"] = "00" * 32
        with self.assertRaisesRegex(ValueError, "measured Compose hash differs"):
            self.verify()

    def test_rejects_missing_measured_compose_event(self):
        self.evidence["event_log"] = []
        with self.assertRaisesRegex(ValueError, "exactly one measured"):
            self.verify()

    def test_rejects_nonempty_wrong_event_digest(self):
        self.evidence["event_log"][0]["digest"] = "00" * 48
        with self.assertRaisesRegex(ValueError, "runtime event digest mismatch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
