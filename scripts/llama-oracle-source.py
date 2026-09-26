#!/usr/bin/env python3
"""Verify that a test oracle uses the current pinned llama.cpp patch queue."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def ordered_patches(patch_dir: Path) -> list[Path]:
    """Bind core, model-support, and generated patches in native preparation order."""
    patches = sorted(patch_dir.glob("*.patch"))
    for expected, patch in enumerate(patches, start=1):
        if not re.fullmatch(rf"{expected:04d}-.+\.patch", patch.name):
            raise RuntimeError(f"invalid top-level patch sequence: {patch.name}")
    for lane, suffix in (
        ("model_support", r"[a-z0-9][a-z0-9.-]*"),
        ("generated", r"family-[a-z0-9.-]+(?:--[a-z0-9.-]+)*"),
    ):
        directory = patch_dir / lane
        if not directory.exists():
            continue
        series = directory / "series"
        if not series.is_file():
            raise RuntimeError(f"{lane} patch directory has no series file")
        names = series.read_text(encoding="utf-8").splitlines()
        if not names or len(names) != len(list(directory.glob("*.patch"))):
            raise RuntimeError(f"{lane} patch series does not cover its patch directory")
        for expected, name in enumerate(names, start=1):
            if not re.fullmatch(rf"{expected:04d}-{suffix}\.patch", name):
                raise RuntimeError(f"invalid {lane} patch sequence: {name}")
            patch = directory / name
            if not patch.is_file():
                raise RuntimeError(f"{lane} patch is missing: {name}")
            patches.append(patch)
    return patches


def patch_digest(patch_dir: Path) -> str:
    """Hash patch names and contents in validated application order."""
    digest = hashlib.sha256()
    for patch in ordered_patches(patch_dir):
        relative_name = patch.relative_to(patch_dir).as_posix()
        file_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        digest.update(f"{relative_name}\n{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()


def prepared_patched_sha(root: Path) -> str:
    """Verify the prepared checkout's upstream pin, patch digest, schema, and clean HEAD."""
    checkout = root / ".deps/llama.cpp"
    prepared_upstream = (checkout / ".mesh-llm-upstream-sha").read_text(encoding="utf-8").strip()
    prepared_patch_digest = (checkout / ".mesh-llm-patch-digest").read_text(encoding="utf-8").strip()
    prepared_patched = (checkout / ".mesh-llm-patched-sha").read_text(encoding="utf-8").strip()
    prepared_schema = (checkout / ".mesh-llm-prepare-schema").read_text(encoding="utf-8").strip()
    upstream = (root / "third_party/llama.cpp/upstream.txt").read_text(encoding="utf-8").strip()
    if prepared_schema != "5" or prepared_upstream != upstream:
        raise RuntimeError("prepared llama.cpp checkout does not match the pinned upstream")
    if prepared_patch_digest != patch_digest(root / "third_party/llama.cpp/patches"):
        raise RuntimeError("prepared llama.cpp checkout does not match the current patch queue")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if prepared_patched != head:
        raise RuntimeError("prepared llama.cpp patched SHA does not match Git HEAD")
    subprocess.run(["git", "-C", str(checkout), "diff-index", "--quiet", "HEAD", "--"],
                   check=True)
    return prepared_patched


def main() -> None:
    """Print the verified patched revision or explain its provenance mismatch."""
    try:
        print(prepared_patched_sha(ROOT))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"pinned llama.cpp oracle source is stale: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
