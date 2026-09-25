#!/usr/bin/env python3
"""Generate the architecture split-serving certification roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
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


def _series_patches(directory: Path) -> list[Path]:
    # Keep this series validation and ordering contract in lockstep with
    # crates/mesh-llm-host-runtime/build.rs::series_patches. Together they
    # define the v2 patch_queue_sha256 embedded in the generated roster.
    if not directory.exists():
        return []
    series = directory / "series"
    if not series.is_file():
        raise RosterError(f"patch directory is missing its series file: {directory}")
    names = [line.rstrip("\r") for line in series.read_text(encoding="utf-8").splitlines()]
    if not names or any(not name for name in names):
        raise RosterError(f"patch series is empty or contains blank entries: {series}")
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise RosterError(f"patch series contains duplicate or unsafe entries: {series}")
    patches = [directory / name for name in names]
    if any(not path.is_file() for path in patches):
        raise RosterError(f"patch series lists a missing patch: {series}")
    actual = {path.name for path in directory.glob("*.patch")}
    if actual != set(names):
        raise RosterError(f"patch series does not exactly cover its directory: {series}")
    return patches


def ordered_patch_queue() -> list[Path]:
    return [
        *sorted(PATCH_DIR.glob("*.patch")),
        *_series_patches(PATCH_DIR / "model_support"),
        *_series_patches(PATCH_DIR / "generated"),
    ]


def patch_queue_sha256() -> str:
    hasher = hashlib.sha256()
    hasher.update(b"mesh-llm-skippy-patch-queue-v2\0")
    patches = ordered_patch_queue()
    hasher.update(struct.pack("<Q", len(patches)))
    for path in patches:
        _frame(hasher, path.relative_to(PATCH_DIR).as_posix().encode())
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


def build_roster(manifest: dict[str, Any]) -> dict[str, Any]:
    """Bind only causal split-certified architectures to this exact native recipe."""
    policy = manifest.get("policy")
    models = manifest.get("models")
    if not isinstance(policy, dict) or not isinstance(models, list):
        raise RosterError("family certification manifest is malformed")
    profiles = policy.get("profiles")
    if not isinstance(profiles, dict):
        raise RosterError("family certification profiles are missing")

    architectures: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise RosterError("family certification model row must be an object")
        profile_name = model.get("profile")
        model_class = model.get("class")
        if model_class in (
            "embedding", "rerank", "encoder_decoder", "ocr",
            "speech_synthesis", "speech_recognition",
        ):
            if profile_name not in ("workload-smoke", "workload-oracle"):
                raise RosterError("non-chat model cannot claim a split-certified profile")
            continue
        if model_class != "causal_generation":
            raise RosterError("family certification row has a missing or unknown workload class")
        if profile_name in ("workload-smoke", "workload-oracle"):
            raise RosterError("causal model cannot use a non-chat workload profile")
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
        architecture = model.get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise RosterError("certified row is missing architecture")
        architectures.add(architecture)

    if not architectures:
        raise RosterError("family manifest produced no split-certified architectures")
    return {
        "schema_version": 2,
        "native_recipe": {
            "llama_upstream_sha": UPSTREAM_PIN.read_text(encoding="utf-8").strip(),
            "skippy_abi": skippy_abi(),
            "patch_queue_sha256": patch_queue_sha256(),
        },
        "architectures": sorted(architectures),
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
