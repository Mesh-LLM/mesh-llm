#!/usr/bin/env python3
"""Select the previous stable release tag for GitHub-generated notes."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable


TARGET_TAG = re.compile(
    r"^v?(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
    r"(?:-[0-9A-Za-z.-]+)?$"
)
STABLE_TAG = re.compile(
    r"^v(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)$"
)


def version_from_match(match: re.Match[str]) -> tuple[int, int, int]:
    return tuple(int(match.group(name)) for name in ("major", "minor", "patch"))


def select_release_notes_base(target: str, tags: Iterable[str]) -> str | None:
    target_match = TARGET_TAG.fullmatch(target.strip())
    if target_match is None:
        raise ValueError(f"invalid release tag: {target}")
    target_version = version_from_match(target_match)

    candidates: list[tuple[tuple[int, int, int], str]] = []
    for raw_tag in tags:
        tag = raw_tag.strip()
        match = STABLE_TAG.fullmatch(tag)
        if match is None:
            continue
        version = version_from_match(match)
        if version < target_version:
            candidates.append((version, tag))

    if not candidates:
        return None
    return max(candidates)[1]


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: select-release-notes-base.py <target-tag>",
            file=sys.stderr,
        )
        return 2

    try:
        selected = select_release_notes_base(sys.argv[1], sys.stdin)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if selected is not None:
        print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
