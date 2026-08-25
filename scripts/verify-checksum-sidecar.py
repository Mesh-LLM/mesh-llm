#!/usr/bin/env python3
"""Require and verify the canonical SHA-256 sidecar for one artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re


SIDECAR_LINE = re.compile(
    r"([0-9a-f]{64}) {2}([^/\\\r\n]+)",
)


def sha256_file(path: Path) -> str:
    """Execute sha256 file operation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(artifact: Path) -> None:
    """Execute verify operation."""
    sidecar = artifact.with_name(f"{artifact.name}.sha256")
    if not sidecar.is_file() or sidecar.stat().st_size == 0:
        raise ValueError(
            f"archive checksum sidecar is missing or empty: {sidecar}"
        )
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(
            "checksum sidecar must contain exactly one canonical line"
        )
    match = SIDECAR_LINE.fullmatch(lines[0])
    if match is None:
        raise ValueError(
            "checksum sidecar must use '<sha256>  <archive-name>' format"
        )
    expected, recorded_name = match.groups()
    if recorded_name != artifact.name:
        raise ValueError(
            f"checksum sidecar names {recorded_name!r}, "
            f"expected {artifact.name!r}"
        )
    actual = sha256_file(artifact)
    if actual != expected.lower():
        raise ValueError(
            f"archive checksum mismatch: {artifact}\n"
            f"  expected: {expected.lower()}\n"
            f"  actual:   {actual}"
        )


def main() -> int:
    """Execute main program logic.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        verify(args.artifact)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
