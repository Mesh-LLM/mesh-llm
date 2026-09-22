#!/usr/bin/env python3
"""Freeze an explicitly trusted same-repository branch/commit for certification."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def resolve(root: Path, ref: str, upstream: str = "") -> dict[str, str]:
    if upstream:
        raise ValueError("mesh_ref cannot be combined with upstream_sha; its existing pin is certified")
    if not ref or ref != ref.strip():
        raise ValueError("mesh_ref must be a branch name or full commit SHA")
    is_sha = re.fullmatch(r"[0-9a-f]{40}", ref) is not None
    branch = ref.removeprefix("refs/heads/")
    if not is_sha:
        if ref.startswith("refs/") and not ref.startswith("refs/heads/"):
            raise ValueError("mesh_ref accepts branches, not pull-request refs or tags")
        git(root, "check-ref-format", "refs/heads/" + branch)
    # Fetch only this repository's branch namespace. A raw SHA must be reachable
    # from one of those branches; GitHub can otherwise serve fork-only objects.
    depth = ["--unshallow"] if git(root, "rev-parse", "--is-shallow-repository") == "true" else []
    git(root, "fetch", "--no-tags", "--prune", *depth, "origin",
        "+refs/heads/*:refs/canary-mesh/*")
    target = ref if is_sha else "refs/canary-mesh/" + branch
    source = git(root, "rev-parse", "--verify", target + "^{commit}")
    if not git(root, "for-each-ref", "--format=%(refname)", "--contains=" + source, "refs/canary-mesh/"):
        raise ValueError("mesh_ref commit is not reachable from a same-repository branch")
    pin = git(root, "show", source + ":third_party/llama.cpp/upstream.txt")
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError("selected MeshLLM revision has an invalid llama.cpp pin")
    return {"mesh_source": source, "upstream": pin, "changed": "false",
            "mode": "pinned-build", "certify": "true"}


def main() -> None:
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("mesh_ref requires an explicit manual dispatch")
    values = resolve(Path.cwd(), os.environ["MESH_REF"], os.environ.get("UPSTREAM", ""))
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(f"Certify-only MeshLLM revision: `{values['mesh_source']}`\n\n"
                     f"Existing llama.cpp pin: `{values['upstream']}`\n")


if __name__ == "__main__":
    main()
