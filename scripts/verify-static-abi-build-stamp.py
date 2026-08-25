#!/usr/bin/env python3
"""Validate the portable llama.cpp static-ABI build-stamp contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


FIELD_NAME = re.compile(r"^[a-z][a-z0-9-]*$")
REQUIRED_FIELDS = (
    "stamp-version",
    "patched-sha",
    "backend",
    "link-mode",
    "toolchain-epoch",
)


class StampError(RuntimeError):
    """Raised when a build stamp does not satisfy the portable ABI contract."""


def parse_args() -> argparse.Namespace:
    """Parse and validate args.

    Returns:
        Parsed result.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stamp", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--link-mode", required=True)
    parser.add_argument("--stamp-version", required=True)
    parser.add_argument("--toolchain-epoch", required=True)
    parser.add_argument("--patched-sha")
    return parser.parse_args()


def parse_stamp(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse and validate stamp.

    Returns:
        Parsed result.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise StampError(f"unable to read static ABI build stamp: {error}") from error

    fields: dict[str, str] = {}
    cmake_arguments: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        key, separator, value = line.partition("=")
        if not separator or not FIELD_NAME.fullmatch(key):
            raise StampError(
                f"static ABI build stamp line {line_number} is malformed",
            )
        if key == "cmake-arg":
            cmake_arguments.append(value)
            continue
        if key in fields:
            raise StampError(
                f"static ABI build stamp repeats singleton field {key!r}",
            )
        fields[key] = value

    missing = [name for name in REQUIRED_FIELDS if not fields.get(name)]
    if missing:
        raise StampError(
            "static ABI build stamp is missing required fields: "
            + ", ".join(missing),
        )
    if not cmake_arguments:
        raise StampError("static ABI build stamp must contain at least one cmake-arg")
    return fields, cmake_arguments


def require_equal(fields: dict[str, str], name: str, expected: str) -> None:
    """Execute require equal operation."""
    actual = fields.get(name)
    if actual != expected:
        raise StampError(
            f"static ABI build stamp {name} mismatch: "
            f"expected {expected!r}, got {actual!r}",
        )


def main() -> int:
    """Execute main program logic.

    Returns:
        Exit code.
    """
    arguments = parse_args()
    try:
        fields, cmake_arguments = parse_stamp(arguments.stamp)
        require_equal(fields, "backend", arguments.backend)
        require_equal(fields, "link-mode", arguments.link_mode)
        require_equal(fields, "stamp-version", arguments.stamp_version)
        require_equal(fields, "toolchain-epoch", arguments.toolchain_epoch)
        if arguments.patched_sha is not None:
            require_equal(fields, "patched-sha", arguments.patched_sha)
    except StampError as error:
        raise SystemExit(str(error)) from error

    print(
        "verified static ABI build stamp: "
        f"backend={fields['backend']} "
        f"cmake_arguments={len(cmake_arguments)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
