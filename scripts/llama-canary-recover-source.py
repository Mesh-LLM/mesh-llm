#!/usr/bin/env python3
"""Save an unverified canary repair for diagnosis after a failed agent handoff."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "core.fsmonitor=false", *args],
    )


def snapshot(root: Path, destination: Path, base: str) -> dict[str, object]:
    head = git(root, "rev-parse", "HEAD").decode().strip()
    destination.mkdir(parents=True, exist_ok=True)
    # The base comparison includes committed, staged, and unstaged edits, even
    # if the agent violated the no-commit instruction. Keep new files separate.
    patch = git(root, "diff", "--binary", "--no-ext-diff", "--no-textconv", base, "--")
    (destination / "tracked.patch").write_bytes(patch)
    names = []
    for name in git(root, "ls-files", "-z", "--others", "--exclude-standard").split(b"\0"):
        if name and not (root / name.decode("utf-8", "surrogateescape")).is_relative_to(destination):
            names.append(name)
    with tarfile.open(destination / "untracked.tar.gz", "w:gz", dereference=False) as archive:
        for raw_name in names:
            name = raw_name.decode("utf-8", "surrogateescape")
            path = root / name
            if path.exists() or path.is_symlink():
                archive.add(path, arcname=name, recursive=False)
    return {"base": base, "head": head, "tracked_patch_bytes": len(patch),
            "untracked_files": len(names)}


def save(root: Path, destination: Path, base: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("recovery base must be a full commit SHA")
    git(root, "cat-file", "-e", f"{base}^{{commit}}")
    manifest = snapshot(root, destination, base)
    nested = root / ".deps" / "llama.cpp"
    if nested.is_dir() and (nested / ".git").exists():
        nested_base = git(nested, "rev-parse", "HEAD").decode().strip()
        manifest["prepared_llama_cpp"] = snapshot(nested, destination / "prepared-llama-cpp", nested_base)
    manifest["verified"] = False
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    save(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3])
