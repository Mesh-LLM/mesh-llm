#!/usr/bin/env python3
"""Generate the exact-artifact split-serving certification roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.json"
DEFAULT_OUTPUT = (
    ROOT
    / "crates"
    / "mesh-llm-host-runtime"
    / "src"
    / "inference"
    / "skippy"
    / "split-certified.json"
)
UPSTREAM_PIN = ROOT / "third_party" / "llama.cpp" / "upstream.txt"
PATCH_DIR = ROOT / "third_party" / "llama.cpp" / "patches"
ABI_SOURCE = ROOT / "crates" / "skippy-ffi" / "src" / "lib.rs"


class RosterError(ValueError):
    """Raised when certification input cannot produce a safe roster."""


def _frame(hasher: Any, value: bytes) -> None:
    hasher.update(struct.pack("<Q", len(value)))
    hasher.update(value)


def patch_queue_sha256() -> str:
    hasher = hashlib.sha256()
    hasher.update(b"mesh-llm-skippy-patch-queue-v1\0")
    patches = sorted(PATCH_DIR.glob("*.patch"))
    hasher.update(struct.pack("<Q", len(patches)))
    for path in patches:
        _frame(hasher, path.name.encode())
        _frame(hasher, path.read_bytes())
    return hasher.hexdigest()


def skippy_abi() -> str:
    values: dict[str, int] = {}
    for line in ABI_SOURCE.read_text(encoding="utf-8").splitlines():
        for name in ("MAJOR", "MINOR", "PATCH"):
            prefix = f"pub const ABI_VERSION_{name}: u32 = "
            if line.startswith(prefix) and line.endswith(";"):
                values[name] = int(line[len(prefix) : -1])
    if set(values) != {"MAJOR", "MINOR", "PATCH"}:
        raise RosterError("could not read the complete Skippy ABI version")
    return f"{values['MAJOR']}.{values['MINOR']}.{values['PATCH']}"


def aggregate_source_sha256(files: list[str], integrity: dict[str, Any]) -> str:
    if not files:
        raise RosterError("certified artifact must contain at least one GGUF file")
    records: list[tuple[int, str]] = []
    for name in files:
        record = integrity.get(name)
        if not isinstance(record, dict):
            raise RosterError(f"missing file_integrity for {name}")
        size = record.get("size_bytes")
        digest = record.get("blob_id")
        if type(size) is not int or size <= 0:
            raise RosterError(f"invalid size_bytes for {name}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RosterError(f"invalid blob_id for {name}")
        records.append((size, digest))
    if len(records) == 1:
        return records[0][1]
    hasher = hashlib.sha256()
    hasher.update(b"mesh-llm-split-gguf-v1\0")
    hasher.update(struct.pack("<Q", len(records)))
    for index, (size, digest) in enumerate(records):
        hasher.update(struct.pack("<Q", index))
        hasher.update(struct.pack("<Q", size))
        hasher.update(digest.encode())
    return hasher.hexdigest()


def build_roster(manifest: dict[str, Any]) -> dict[str, Any]:
    policy = manifest.get("policy")
    models = manifest.get("models")
    if not isinstance(policy, dict) or not isinstance(models, list):
        raise RosterError("family certification manifest is malformed")
    profiles = policy.get("profiles")
    if not isinstance(profiles, dict):
        raise RosterError("family certification profiles are missing")

    entries = []
    seen_sources: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise RosterError("family certification model row must be an object")
        profile_name = model.get("profile")
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict) or profile.get("status") != "certified":
            continue
        required_lanes = profile.get("required_lanes")
        if not isinstance(required_lanes, list) or not {
            "single-step",
            "chain",
            "state-handoff",
        }.issubset(required_lanes):
            continue
        family = model.get("family")
        artifact = model.get("artifact")
        if not isinstance(family, str) or not isinstance(artifact, dict):
            raise RosterError("certified row is missing family or artifact")
        files = artifact.get("files")
        integrity = artifact.get("file_integrity")
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise RosterError(f"{family}: artifact.files must be strings")
        if not isinstance(integrity, dict):
            raise RosterError(f"{family}: artifact.file_integrity is missing")
        source_sha = aggregate_source_sha256(files, integrity)
        if source_sha in seen_sources:
            raise RosterError(f"duplicate certified source identity: {source_sha}")
        seen_sources.add(source_sha)
        entries.append(
            {
                "family": family,
                "repo": artifact.get("repo"),
                "revision": artifact.get("revision"),
                "files": files,
                "package_kind": "content-addressed-direct-gguf",
                "source_model_sha256": source_sha,
            }
        )

    entries.sort(key=lambda entry: (entry["family"], entry["source_model_sha256"]))
    if not entries:
        raise RosterError("family manifest produced no split-certified models")
    return {
        "schema_version": 1,
        "native_recipe": {
            "llama_upstream_sha": UPSTREAM_PIN.read_text(encoding="utf-8").strip(),
            "skippy_abi": skippy_abi(),
            "patch_queue_sha256": patch_queue_sha256(),
        },
        "models": entries,
    }


def encoded_roster(manifest_path: Path) -> bytes:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (json.dumps(build_roster(manifest), indent=2) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = encoded_roster(args.manifest)
    if args.check:
        actual = args.output.read_bytes() if args.output.exists() else b""
        if actual != expected:
            print(
                f"{args.output} is stale; regenerate it with scripts/generate-split-certified.py",
                file=sys.stderr,
            )
            return 1
        print(f"split certification roster is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(expected)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, RosterError) as error:
        print(f"split certification roster generation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
