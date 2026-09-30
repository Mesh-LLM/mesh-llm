#!/usr/bin/env python3
"""Verify a dstack TDX quote for one Mesh iroh peer before pinning a request.

This is an independent, manual verifier. It does not turn on TEE-only routing.
Install dcap-qvl==0.6.5 in a virtual environment before using it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import secrets
import sys
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


FORMAT = "mesh-tee-endpoint-v1"
REPORT_DATA_DOMAIN = b"mesh-tee-endpoint-v1\x00"
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_QUOTE_BYTES = 64 * 1024
MEASUREMENTS = ("mr_td", "rt_mr0", "rt_mr1", "rt_mr2", "rt_mr3")
POLICY_FIELDS = frozenset((*MEASUREMENTS, "app_compose_sha256"))


def hex_bytes(value: object, byte_count: int, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be hexadecimal text")
    if re.fullmatch(rf"[0-9a-fA-F]{{{byte_count * 2}}}", value) is None:
        raise ValueError(f"{label} must be {byte_count} bytes of hexadecimal text")
    return bytes.fromhex(value)


def load_policy(path: Path) -> dict[str, bytes]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != POLICY_FIELDS:
        fields = ", ".join(sorted(POLICY_FIELDS))
        raise ValueError(f"policy must contain exactly: {fields}")
    return {
        field: hex_bytes(raw[field], 32 if field == "app_compose_sha256" else 48, field)
        for field in POLICY_FIELDS
    }


def report_data(nonce: bytes, endpoint_id: bytes) -> bytes:
    return hashlib.sha256(REPORT_DATA_DOMAIN + nonce + endpoint_id).digest() + bytes(32)


def raw_quote(evidence: dict[str, Any], expected_nonce: bytes) -> bytes:
    if evidence.get("format") != FORMAT:
        raise ValueError("unsupported evidence format")
    if hex_bytes(evidence.get("nonce"), 32, "nonce") != expected_nonce:
        raise ValueError("evidence nonce differs from the verifier challenge")
    quote_hex = evidence.get("quote")
    if not isinstance(quote_hex, str) or len(quote_hex) > MAX_QUOTE_BYTES * 2:
        raise ValueError("quote is missing or too large")
    if re.fullmatch(r"[0-9a-fA-F]+", quote_hex) is None or len(quote_hex) % 2:
        raise ValueError("quote is not hexadecimal")
    return bytes.fromhex(quote_hex)


def replay_rtmr3(events: object, expected_compose_hash: bytes) -> bytes:
    if isinstance(events, str):
        events = json.loads(events)
    if not isinstance(events, list) or len(events) > 256:
        raise ValueError("runtime event log must be a bounded list")
    register = bytes(48)
    compose_events = 0
    for event in events:
        if not isinstance(event, dict) or type(event.get("imr")) is not int:
            raise ValueError("invalid runtime event")
        if event["imr"] != 3:
            continue
        if event.get("event_type") != 0x08000001 or event.get("version", 1) != 1:
            raise ValueError("unsupported RTMR3 event format")
        name = event.get("event")
        if not isinstance(name, str):
            raise ValueError("runtime event name is missing")
        payload = event.get("event_payload")
        if not isinstance(payload, str):
            raise ValueError("runtime event payload is missing")
        payload_bytes = bytes.fromhex(payload.removeprefix("0x"))
        if name == "compose-hash":
            compose_events += 1
            if payload_bytes != expected_compose_hash:
                raise ValueError("measured Compose hash differs from policy")
        digest = hashlib.sha384(
            event["event_type"].to_bytes(4, "little")
            + b":" + name.encode() + b":" + payload_bytes
        ).digest()
        advertised = event.get("digest")
        if advertised is not None:
            if not isinstance(advertised, str):
                raise ValueError("runtime event digest is invalid")
            if advertised and bytes.fromhex(advertised.removeprefix("0x")) != digest:
                raise ValueError("runtime event digest mismatch")
        register = hashlib.sha384(register + digest).digest()
    if compose_events != 1:
        raise ValueError("expected exactly one measured Compose hash")
    return register


def verify_evidence(
    evidence: dict[str, Any],
    expected_nonce: bytes,
    expected_peer: bytes,
    policy: dict[str, bytes],
    collateral: object,
    qvl: Any,
    now: int,
) -> dict[str, str]:
    quote_bytes = raw_quote(evidence, expected_nonce)
    # QVL checks the Intel trust chain and collateral freshness at `now`.
    verdict = qvl.verify(quote_bytes, collateral, now)
    if verdict.status != "UpToDate" or verdict.advisory_ids:
        raise ValueError("TDX TCB is not up to date and free of advisories")
    quote = qvl.Quote.parse(quote_bytes)
    if not quote.is_tdx():
        raise ValueError("evidence is not an Intel TDX quote")
    if int.from_bytes(quote.report.td_attributes, "little") & 1:
        raise ValueError("TDX debug mode is enabled")
    endpoint_id = hex_bytes(evidence.get("endpoint_id"), 32, "endpoint_id")
    if endpoint_id != expected_peer:
        raise ValueError("attested endpoint differs from the selected Mesh peer")
    if quote.report.report_data != report_data(expected_nonce, expected_peer):
        raise ValueError("quote does not bind the challenge to the Mesh peer")
    for field in MEASUREMENTS:
        if getattr(quote.report, field) != policy[field]:
            raise ValueError(f"TDX {field} differs from approved policy")
    app_compose = evidence.get("app_compose")
    if not isinstance(app_compose, str):
        raise ValueError("app_compose is missing")
    compose_hash = hashlib.sha256(app_compose.encode()).digest()
    if compose_hash != policy["app_compose_sha256"]:
        raise ValueError("deployed application Compose differs from approved policy")
    if hex_bytes(evidence.get("compose_hash"), 32, "compose_hash") != compose_hash:
        raise ValueError("advertised Compose hash differs from application Compose")
    if replay_rtmr3(evidence.get("event_log"), compose_hash) != quote.report.rt_mr3:
        raise ValueError("runtime event log does not match quoted RTMR3")
    return {
        "status": "verified",
        "tee": "intel-tdx",
        "endpoint_id": endpoint_id.hex(),
        "nonce": expected_nonce.hex(),
        "tcb_status": verdict.status,
        "app_compose_sha256": compose_hash.hex(),
    }


def read_evidence(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(data) > MAX_EVIDENCE_BYTES:
        raise ValueError("evidence file is too large")
    evidence = json.loads(data)
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a JSON object")
    return evidence


def challenge(url: str, nonce: bytes) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        raise ValueError("attestation URL must use HTTP or HTTPS")
    request_url = url.rstrip("/") + "/attest?" + urlencode({"nonce": nonce.hex()})
    with urlopen(request_url, timeout=30) as response:
        data = response.read(MAX_EVIDENCE_BYTES + 1)
    if len(data) > MAX_EVIDENCE_BYTES:
        raise ValueError("attestation response is too large")
    evidence = json.loads(data)
    if not isinstance(evidence, dict):
        raise ValueError("attestation response must be a JSON object")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument(
        "--peer", required=True, help="selected iroh EndpointId, 32 bytes hex"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url", help="live attestor base URL; creates a fresh challenge"
    )
    source.add_argument(
        "--evidence", type=Path, help="saved evidence; historical inspection only"
    )
    parser.add_argument(
        "--nonce", help="required with --evidence; 32-byte original challenge"
    )
    args = parser.parse_args()
    if (args.evidence is not None) != (args.nonce is not None):
        parser.error("--nonce is required with --evidence and forbidden with --url")
    try:
        import dcap_qvl

        policy = load_policy(args.policy)
        expected_peer = hex_bytes(args.peer, 32, "peer")
        nonce = (
            hex_bytes(args.nonce, 32, "nonce")
            if args.nonce is not None
            else secrets.token_bytes(32)
        )
        evidence = (
            read_evidence(args.evidence)
            if args.evidence is not None
            else challenge(args.url, nonce)
        )
        quote_bytes = raw_quote(evidence, nonce)
        # Intel PCS supplies Intel-signed collateral; QVL verifies its signatures.
        collateral = asyncio.run(dcap_qvl.get_collateral_from_pcs(quote_bytes))
        result = verify_evidence(
            evidence,
            nonce,
            expected_peer,
            policy,
            collateral,
            dcap_qvl,
            int(time.time()),
        )
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(1, f"TEE verification failed: {error}\n")
    output = {**result, "fresh_challenge": args.url is not None}
    json.dump(output, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
