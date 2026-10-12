#!/usr/bin/env python3
"""Verify the standalone Skippy CLI producer bytes and embedded build contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys


SHA = re.compile(r"[0-9a-f]{40}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path, source_sha: str) -> dict[str, object]:
    if not SHA.fullmatch(source_sha):
        raise ValueError("source SHA must be a full lowercase Git commit ID")
    names = [name for name in ("skippy", "skippy.exe") if (directory / name).is_file()]
    if len(names) != 1:
        raise ValueError("CLI input must contain exactly one Skippy executable")
    name = names[0]
    binary = directory / name
    digest = sha256(binary)
    sidecar = (directory / f"{name}.sha256").read_text(encoding="utf-8").strip()
    if sidecar != f"{digest}  {name}":
        raise ValueError("Skippy CLI checksum sidecar does not match executable")
    # GitHub artifact download does not preserve POSIX executable mode.
    if name == "skippy":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    imports = json.loads((directory / "host-imports.json").read_text(encoding="utf-8"))
    if (
        not isinstance(imports, dict)
        or imports.get("binary") != name
        or imports.get("binary_sha256") != digest
        or imports.get("policy") != "mesh-llm-dynamic-host-v2"
        or imports.get("rejected_imports") != []
        or not isinstance(imports.get("imports"), list)
        or not all(isinstance(item, str) for item in imports["imports"])
        or imports.get("format") not in {"elf", "macho", "pe"}
    ):
        raise ValueError("Skippy CLI import-policy report is missing, foreign, or rejected")
    recorded = json.loads((directory / "build-contract.json").read_text(encoding="utf-8-sig"))
    actual = json.loads(
        subprocess.run(
            [str(binary.resolve()), "build-contract"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )
    if recorded != actual:
        raise ValueError("recorded Skippy build contract differs from executable")
    if not isinstance(actual, dict) or actual.get("schema_version") != 1 or actual.get("product") != "skippy":
        raise ValueError("unsupported Skippy build contract")
    if actual.get("source_sha") != source_sha:
        raise ValueError("Skippy CLI was built from a different source commit")
    expected_format = {"linux": "elf", "macos": "macho", "windows": "pe"}.get(actual.get("os"))
    if imports["format"] != expected_format or (name.endswith(".exe") != (expected_format == "pe")):
        raise ValueError("Skippy CLI import format differs from executable target")
    for field in ("product_version", "runtime_release", "skippy_abi", "os", "architecture"):
        if not isinstance(actual.get(field), str) or not actual[field]:
            raise ValueError(f"Skippy build contract is missing {field}")
    version = subprocess.run(
        [str(binary.resolve()), "--version"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    if version != f"skippy {actual['product_version']}":
        raise ValueError("Skippy CLI version differs from embedded build contract")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    try:
        verify(args.directory, args.source_sha)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Skippy CLI input verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
