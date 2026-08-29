from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import secrets
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "llama-upstream-canary.yml"
NOTIFY = ROOT / "scripts" / "llama-canary-buzz-notify.py"


def _load_notify():
    spec = importlib.util.spec_from_file_location("llama_canary_buzz_notify", NOTIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step_block(workflow: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    start = workflow.index(marker)
    end = workflow.find("\n      - name: ", start + len(marker))
    return workflow[start:] if end == -1 else workflow[start:end]


class LlamaCanaryBuzzNotifyWorkflowTests(unittest.TestCase):
    def test_notify_step_is_best_effort_and_secret_scoped(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        notify = _step_block(workflow, "Notify Buzz channel")
        # Notification is telemetry: it must never fail the canary run.
        self.assertIn("continue-on-error: true", notify)
        # The notify key arrives as a secret env var, never inline.
        self.assertIn("BUZZ_NOTIFY_KEY: ${{ secrets.BUZZ_RELAY_KEY }}", notify)
        self.assertIn("BUZZ_RELAY_AUTH_TAG: ${{ secrets.BUZZ_RELAY_AUTH_TAG }}", notify)
        # Channel is a bounded selector with a checked-in default (#mesh-dev).
        self.assertIn("BUZZ_CHANNEL: ${{ vars.LLAMA_CANARY_BUZZ_CHANNEL ||", notify)
        self.assertIn("a9ea5982-8d74-407f-b82e-6b55c69d411f", notify)
        # A missing key skips quietly instead of failing.
        self.assertIn('[[ -z "${BUZZ_NOTIFY_KEY}" ]]', notify)
        self.assertIn("python3 scripts/llama-canary-buzz-notify.py", notify)
        # Failure downgrades to a warning, not an error.
        self.assertIn("best-effort, run continues", notify)

    def test_notify_fires_for_both_outcomes(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        notify = _step_block(workflow, "Notify Buzz channel")
        self.assertIn("steps.outcome.outputs.outcome == 'merge'", notify)
        self.assertIn("steps.outcome.outputs.outcome == 'stuck'", notify)


class LlamaCanaryBuzzNotifyScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notify = _load_notify()

    def test_bip340_signature_roundtrips(self) -> None:
        # Sign with the script's signer, verify with an independent even-y
        # lift reconstruction — proves key handling and nonce parity logic.
        priv = secrets.token_bytes(32)
        msg = secrets.token_bytes(32)
        sig = self.notify._sign_bip340(priv, msg, secrets.token_bytes(32))
        self.assertEqual(64, len(sig))

        G = self.notify.G
        P = self.notify.P
        N = self.notify.N
        pub = self.notify._pmul(int.from_bytes(priv, "big"), G)
        pub_even = pub if pub[1] % 2 == 0 else (pub[0], (P - pub[1]) % P)
        rx = int.from_bytes(sig[:32], "big")
        s = int.from_bytes(sig[32:], "big")
        e = int.from_bytes(
            self.notify._tagged(
                "BIP0340/challenge",
                sig[:32] + pub_even[0].to_bytes(32, "big") + msg,
            ),
            "big",
        ) % N
        ep = self.notify._pmul(e, pub_even)
        neg_ep = (ep[0], (P - ep[1]) % P)
        r_check = self.notify._padd(self.notify._pmul(s, G), neg_ep)
        self.assertEqual(rx, r_check[0])
        self.assertEqual(0, r_check[1] % 2)

    def test_event_id_hash_matches_canonical_serialization(self) -> None:
        priv = secrets.token_bytes(32)
        point = self.notify._pmul(int.from_bytes(priv, "big"), self.notify.G)
        pubkey = point[0].to_bytes(32, "big").hex()
        ev = self.notify._event(pubkey, 9, [["h", "test-channel"]], "hello", priv)
        import hashlib

        serialized = json.dumps(
            [0, ev["pubkey"], ev["created_at"], ev["kind"], ev["tags"], ev["content"]],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.assertEqual(hashlib.sha256(serialized.encode()).hexdigest(), ev["id"])
        self.assertEqual(128, len(ev["sig"]))

    def test_decode_priv_accepts_hex_and_nsec(self) -> None:
        raw = secrets.token_bytes(32)
        self.assertEqual(raw, self.notify._decode_priv(raw.hex()))
        # nsec roundtrip via proper bech32 (5-bit groups, padding, checksum).
        charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

        def bech32_polymod(values):
            gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
            chk = 1
            for value in values:
                top = chk >> 25
                chk = (chk & 0x1FFFFFF) << 5 ^ value
                for i in range(5):
                    chk ^= gen[i] if ((top >> i) & 1) else 0
            return chk

        acc = 0
        bits = 0
        data_bits = []
        for byte in raw:
            acc = (acc << 8) | byte
            bits += 8
            while bits >= 5:
                bits -= 5
                data_bits.append((acc >> bits) & 31)
        if bits:
            data_bits.append((acc << (5 - bits)) & 31)
        values = [ord(c) - 33 for c in "np"] + data_bits
        checksum = bech32_polymod([0] + values + [0, 0, 0, 0, 0, 0]) ^ 1
        polymod = bech32_polymod(values + [0, 0, 0, 0, 0, 0])
        checksum_values = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
        nsec = "nsec1" + "".join(charset[v] for v in data_bits) + "".join(
            charset[v] for v in checksum_values
        )
        self.assertEqual(raw, self.notify._decode_priv(nsec))

    def test_script_never_prints_the_key(self) -> None:
        source = NOTIFY.read_text(encoding="utf-8")
        # No echo/print of the key variable; only presence checks.
        for banned in (
            'print(key)',
            'print(priv',
            'echo "$key"',
            'echo "$BUZZ_NOTIFY_KEY"',
            'stderr=key',
        ):
            self.assertNotIn(banned, source)


if __name__ == "__main__":
    unittest.main()
