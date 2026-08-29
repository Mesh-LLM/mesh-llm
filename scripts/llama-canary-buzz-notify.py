#!/usr/bin/env python3
"""Post the llama.cpp canary outcome to a Buzz channel over plain HTTP.

Usage:
  llama-canary-buzz-notify.py <channel-uuid> <content-file>

Reads from the environment:
  BUZZ_RELAY_HTTP      relay base URL (default https://meshllm.communities.buzz.xyz)
  BUZZ_NOTIFY_KEY      Nostr private key (hex or nsec) — CI secret BUZZ_RELAY_KEY
  BUZZ_RELAY_AUTH_TAG  optional NIP-OA owner-attestation tag JSON (CI secret
                       BUZZ_RELAY_AUTH_TAG) — passes relay membership when the
                       notify key's owner is a relay member. Direct relay
                       membership for the notify key works without it.

Signs a kind:9 channel message plus a NIP-98 kind:27235 auth event (BIP-340,
stdlib only — no third-party deps on the family-certify runner) and POSTs the
bare event JSON to <relay>/events with `Authorization: Nostr <base64>`.

The transport is curl, not urllib: the relay sits behind Cloudflare bot
management which rejects Python's TLS fingerprint with error 1010; curl's
passes. Callers must not fail the canary run on a non-zero exit — notification
is best-effort telemetry, never a gate. The private key is never echoed or
logged; on failure only the HTTP status and the relay's error body are printed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time

P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (GX, GY)
INF = (0, 0)


def _padd(a, b):
    if a == INF:
        return b
    if b == INF:
        return a
    x1, y1 = a
    x2, y2 = b
    if x1 == x2 and (y1 + y2) % P == 0:
        return INF
    if a == b:
        slope = (3 * x1 * x1) * pow(2 * y1, P - 2, P) % P
    else:
        slope = (y2 - y1) * pow(x2 - x1, P - 2, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return (x3, (slope * (x1 - x3) - y1) % P)


def _pmul(k, pt):
    result = INF
    while k:
        if k & 1:
            result = _padd(result, pt)
        pt = _padd(pt, pt)
        k >>= 1
    return result


def _tagged(tag, msg):
    t = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(t + t + msg).digest()


def _sign_bip340(priv, msg32, aux):
    # BIP-340: sign with the even-y scalar; nonce forced to even-y R.
    d = int.from_bytes(priv, "big")
    point = _pmul(d, G)
    if point[1] % 2 != 0:
        d = N - d
        point = (point[0], (P - point[1]) % P)
    px = point[0].to_bytes(32, "big")
    t = (d ^ int.from_bytes(_tagged("BIP0340/aux", aux), "big")).to_bytes(32, "big")
    k0 = int.from_bytes(_tagged("BIP0340/nonce", t + px + msg32), "big") % N
    r = _pmul(k0, G)
    k = k0 if r[1] % 2 == 0 else N - k0
    e = int.from_bytes(_tagged("BIP0340/challenge", r[0].to_bytes(32, "big") + px + msg32), "big") % N
    return r[0].to_bytes(32, "big") + ((k + e * d) % N).to_bytes(32, "big")


def _decode_priv(value):
    value = value.strip()
    if value.startswith("nsec"):
        charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
        data = value[5:-6]  # strip hrp and 6-char bech32 checksum
        acc = 0
        bits = 0
        out = []
        for char in data:
            acc = (acc << 5) | charset.index(char)
            bits += 5
            if bits >= 8:
                bits -= 8
                out.append((acc >> bits) & 255)
        return bytes(out)
    return bytes.fromhex(value)


def _event(pubkey, kind, tags, content, priv):
    ev = {
        "pubkey": pubkey,
        "created_at": int(time.time()),
        "kind": kind,
        "tags": tags,
        "content": content,
    }
    serialized = json.dumps(
        [0, ev["pubkey"], ev["created_at"], ev["kind"], ev["tags"], ev["content"]],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    ev["id"] = hashlib.sha256(serialized.encode()).hexdigest()
    ev["sig"] = _sign_bip340(priv, bytes.fromhex(ev["id"]), secrets.token_bytes(32)).hex()
    return ev


def main():
    if len(sys.argv) != 3:
        print("usage: llama-canary-buzz-notify.py <channel-uuid> <content-file>", file=sys.stderr)
        return 1
    channel, content_file = sys.argv[1], sys.argv[2]
    relay = os.environ.get("BUZZ_RELAY_HTTP", "https://meshllm.communities.buzz.xyz")
    key = os.environ.get("BUZZ_NOTIFY_KEY")
    if not key:
        print("missing BUZZ_NOTIFY_KEY (CI: set secret BUZZ_RELAY_KEY on the runner)", file=sys.stderr)
        return 1
    priv = _decode_priv(key)
    point = _pmul(int.from_bytes(priv, "big"), G)
    # Nostr pubkeys are the x-only form; the even-y lift happens at verify.
    pubkey = point[0].to_bytes(32, "big").hex()
    with open(content_file, encoding="utf-8") as handle:
        content = handle.read()

    message = _event(pubkey, 9, [["h", channel]], content, priv)
    body = json.dumps(message, separators=(",", ":")).encode()
    auth = _event(
        pubkey,
        27235,
        [["u", relay + "/events"], ["method", "POST"], ["payload", hashlib.sha256(body).hexdigest()]],
        "",
        priv,
    )
    token = base64.b64encode(json.dumps(auth, separators=(",", ":")).encode()).decode()
    result = subprocess.run(
        [
            "curl", "-sS", "-m", "30", "-X", "POST", relay + "/events",
            "-H", "Content-Type: application/json",
            "-H", "Authorization: Nostr " + token,
            "-H", "x-auth-tag: " + (os.environ.get("BUZZ_RELAY_AUTH_TAG") or os.environ.get("BUZZ_AUTH_TAG", "")),
            "--data-binary", "@-",
            "-w", "\n%{http_code}",
        ],
        input=body,
        capture_output=True,
    )
    out = result.stdout.decode(errors="replace").strip()
    print(out[:400])
    lines = out.rsplit("\n", 1)
    status = lines[-1] if lines else ""
    accepted = status == "200" and '"accepted":true' in out
    if not accepted:
        print("buzz notify: relay did not accept the event (status " + status + ")", file=sys.stderr)
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
