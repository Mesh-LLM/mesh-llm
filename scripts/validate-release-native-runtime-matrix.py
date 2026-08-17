#!/usr/bin/env python3
"""Validate release binary bundles against native-runtimes.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any


TARGET_TRIPLES = {
    "aarch64-apple-darwin": ("macos", "aarch64"),
    "x86_64-apple-darwin": ("macos", "x86_64"),
    "x86_64-unknown-linux-gnu": ("linux", "x86_64"),
    "aarch64-unknown-linux-gnu": ("linux", "aarch64"),
    "x86_64-pc-windows-msvc": ("windows", "x86_64"),
}


@dataclass(frozen=True, order=True)
class RuntimeTarget:
    os: str
    arch: str
    backend: str
    cuda_major: int | None = None

    def label(self) -> str:
        if self.backend == "cuda" and self.cuda_major is not None:
            return f"{self.os}/{self.arch}/cuda{self.cuda_major}"
        return f"{self.os}/{self.arch}/{self.backend}"


def default_backend(os_name: str, arch: str) -> str:
    if os_name == "macos" and arch == "aarch64":
        return "metal"
    return "cpu"


def binary_target_from_asset(asset_name: str) -> RuntimeTarget | None:
    name = os.path.basename(asset_name)
    if not name.startswith("mesh-llm-"):
        return None
    if name.endswith(".sha256"):
        return None
    if not (name.endswith(".tar.gz") or name.endswith(".zip")):
        return None

    for triple, (os_name, arch) in TARGET_TRIPLES.items():
        marker = f"-{triple}"
        if marker not in name:
            continue
        suffix = name.split(marker, 1)[1]
        suffix = suffix.removesuffix(".tar.gz").removesuffix(".zip")
        return target_from_suffix(os_name, arch, suffix)
    return None


def target_from_suffix(os_name: str, arch: str, suffix: str) -> RuntimeTarget:
    if suffix == "":
        return RuntimeTarget(os_name, arch, default_backend(os_name, arch))
    if suffix.startswith("-cuda"):
        match = re.fullmatch(r"-cuda(?:-(\d+))?", suffix)
        if not match:
            raise ValueError(f"unsupported CUDA release suffix: {suffix}")
        major = int(match.group(1)) if match.group(1) else None
        return RuntimeTarget(os_name, arch, "cuda", major)
    return RuntimeTarget(os_name, arch, suffix.removeprefix("-"))


def native_target_from_artifact(artifact: dict[str, Any]) -> RuntimeTarget | None:
    platform = artifact.get("platform")
    backend = artifact.get("backend")
    if not isinstance(platform, dict) or not isinstance(backend, dict):
        return None
    os_name = platform.get("os")
    arch = platform.get("arch")
    kind = backend.get("kind")
    if not all(isinstance(value, str) for value in (os_name, arch, kind)):
        return None
    cuda_major = None
    if kind == "cuda":
        cuda = backend.get("cuda")
        if isinstance(cuda, dict) and isinstance(cuda.get("toolkit_major"), int):
            cuda_major = cuda["toolkit_major"]
    return RuntimeTarget(os_name, arch, kind, cuda_major)


def native_target_matches(required: RuntimeTarget, candidate: RuntimeTarget) -> bool:
    if (required.os, required.arch, required.backend) != (
        candidate.os,
        candidate.arch,
        candidate.backend,
    ):
        return False
    if required.backend == "cuda" and required.cuda_major is not None:
        return candidate.cuda_major == required.cuda_major
    return True


def target_from_label(label: str) -> RuntimeTarget:
    parts = label.split("/")
    if len(parts) != 3:
        raise ValueError(f"expected target label as os/arch/backend, got {label!r}")
    os_name, arch, backend = parts
    match = re.fullmatch(r"cuda(\d+)", backend)
    if match:
        return RuntimeTarget(os_name, arch, "cuda", int(match.group(1)))
    return RuntimeTarget(os_name, arch, backend)


def required_targets_from_assets(asset_names: list[str]) -> set[RuntimeTarget]:
    return {
        target
        for asset_name in asset_names
        if (target := binary_target_from_asset(asset_name)) is not None
    }


def find_matrix_violations(
    asset_names: list[str],
    manifest: dict[str, Any],
    required_targets: set[RuntimeTarget] | None = None,
) -> list[str]:
    if required_targets is None:
        required_targets = required_targets_from_assets(asset_names)
    native_targets = [
        target
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict)
        if (target := native_target_from_artifact(artifact)) is not None
    ]
    violations = []
    for required in sorted(required_targets):
        if not any(native_target_matches(required, candidate) for candidate in native_targets):
            violations.append(
                f"missing native runtime for binary target {required.label()}"
            )
    return violations


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate release binary bundle targets against native-runtimes.json."
    )
    parser.add_argument("--manifest", required=True, help="Path to native-runtimes.json")
    parser.add_argument(
        "--required-target",
        action="append",
        default=[],
        help=(
            "Native runtime target that must be present in the manifest, "
            "formatted as os/arch/backend, for example linux/aarch64/cuda13. "
            "When omitted, targets are inferred from release binary asset names."
        ),
    )
    parser.add_argument("assets", nargs="*", help="Release asset paths or names")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.required_target and not args.assets:
        print(
            "release matrix error: assets are required unless --required-target is provided",
            file=sys.stderr,
        )
        return 2
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        required_targets = {
            target_from_label(label) for label in args.required_target
        } or None
    except ValueError as error:
        print(f"release matrix error: {error}", file=sys.stderr)
        return 2
    violations = find_matrix_violations(args.assets, manifest, required_targets)
    if violations:
        for violation in violations:
            print(f"release matrix error: {violation}", file=sys.stderr)
        return 1
    print("release native runtime matrix is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
