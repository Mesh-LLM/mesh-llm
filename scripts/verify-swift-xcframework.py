#!/usr/bin/env python3
"""Verify the immutable Swift XCFramework slice and architecture contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import plistlib
import subprocess
import sys
from typing import Any


PlatformKey = tuple[str, str]

EXPECTED_ARCHITECTURES: dict[str, dict[PlatformKey, frozenset[str]]] = {
    "host-only": {
        ("macos", ""): frozenset({"arm64"}),
    },
    "full": {
        ("ios", ""): frozenset({"arm64"}),
        ("ios", "maccatalyst"): frozenset({"arm64", "x86_64"}),
        ("ios", "simulator"): frozenset({"arm64", "x86_64"}),
        ("macos", ""): frozenset({"arm64", "x86_64"}),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an XCFramework's declared architectures, binary slices, "
            "and versioned macOS framework layout."
        ),
    )
    parser.add_argument("xcframework", type=Path)
    parser.add_argument("mode", nargs="?", choices=sorted(EXPECTED_ARCHITECTURES))
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def require_safe_component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"XCFramework {field} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        fail(f"XCFramework {field} must be one safe path component: {value!r}")
    return value


def platform_key(library: dict[str, Any]) -> PlatformKey:
    platform = library.get("SupportedPlatform")
    variant = library.get("SupportedPlatformVariant", "")
    if not isinstance(platform, str) or not platform:
        fail(f"invalid SupportedPlatform in XCFramework entry: {library!r}")
    if not isinstance(variant, str):
        fail(f"invalid SupportedPlatformVariant in XCFramework entry: {library!r}")
    return platform, variant


def declared_architectures(
    library: dict[str, Any],
    key: PlatformKey,
) -> frozenset[str]:
    architectures = library.get("SupportedArchitectures")
    if not isinstance(architectures, list) or not architectures:
        fail(f"XCFramework slice {key!r} must declare SupportedArchitectures")
    if not all(
        isinstance(architecture, str) and architecture
        for architecture in architectures
    ):
        fail(f"XCFramework slice {key!r} has invalid SupportedArchitectures")
    declared = frozenset(architectures)
    if len(declared) != len(architectures):
        fail(f"XCFramework slice {key!r} declares duplicate architectures")
    return declared


def framework_path(
    xcframework: Path,
    library: dict[str, Any],
) -> Path:
    identifier = require_safe_component(
        library.get("LibraryIdentifier"),
        "LibraryIdentifier",
    )
    library_path = require_safe_component(library.get("LibraryPath"), "LibraryPath")
    if not library_path.endswith(".framework"):
        fail(f"XCFramework LibraryPath must name a framework: {library_path!r}")
    framework = xcframework / identifier / library_path
    resolved_root = xcframework.resolve()
    resolved_framework = framework.resolve()
    if resolved_root not in resolved_framework.parents:
        fail(f"XCFramework library escapes its root: {framework}")
    if not framework.is_dir():
        fail(f"XCFramework library is missing: {framework}")
    return framework


def framework_binary(framework: Path) -> Path:
    name = framework.stem
    binary = framework / name
    if not binary.exists() or not binary.is_file():
        fail(f"XCFramework binary is missing: {binary}")
    return binary


def verify_macos_layout(framework: Path) -> None:
    name = framework.stem
    expected_symlinks = {
        "Versions/Current": "A",
        name: f"Versions/Current/{name}",
        "Headers": "Versions/Current/Headers",
        "Modules": "Versions/Current/Modules",
        "Resources": "Versions/Current/Resources",
    }
    for relative, target in expected_symlinks.items():
        path = framework / relative
        if not path.is_symlink():
            fail(f"macOS framework is not versioned; missing symlink: {path}")
        actual = os.readlink(path)
        if actual != target:
            fail(f"unexpected symlink target for {path}: {actual!r} != {target!r}")

    required_paths = [
        framework / "Versions" / "A" / name,
        framework / "Versions" / "A" / "Headers",
        framework / "Versions" / "A" / "Modules" / "module.modulemap",
        framework / "Versions" / "A" / "Resources" / "Info.plist",
        framework / "Versions" / "A" / "Resources" / "PrivacyInfo.xcprivacy",
    ]
    for path in required_paths:
        if not path.exists():
            fail(f"macOS framework versioned layout is incomplete: {path}")


def lipo_architectures(binary: Path) -> frozenset[str]:
    lipo = os.environ.get("LIPO", "lipo")
    try:
        result = subprocess.run(
            [lipo, "-archs", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"failed to inspect XCFramework binary with lipo: {binary}: {error}")
    architectures = result.stdout.strip().split()
    if not architectures:
        fail(f"lipo reported no architectures for XCFramework binary: {binary}")
    return frozenset(architectures)


def verify_xcframework(xcframework: Path, mode: str | None) -> None:
    info_path = xcframework / "Info.plist"
    if not xcframework.is_dir() or not info_path.is_file():
        fail(f"XCFramework or Info.plist is missing: {xcframework}")
    with info_path.open("rb") as handle:
        info = plistlib.load(handle)

    libraries = info.get("AvailableLibraries")
    if not isinstance(libraries, list) or not libraries:
        fail("XCFramework AvailableLibraries must be a non-empty array")

    entries: dict[PlatformKey, dict[str, Any]] = {}
    for library in libraries:
        if not isinstance(library, dict):
            fail(f"invalid XCFramework library entry: {library!r}")
        key = platform_key(library)
        if key in entries:
            fail(f"XCFramework contains a duplicate platform slice: {key!r}")
        entries[key] = library

    if ("macos", "") not in entries:
        fail("XCFramework does not contain a macOS framework slice")

    expected = EXPECTED_ARCHITECTURES.get(mode) if mode else None
    if expected is not None and entries.keys() != expected.keys():
        fail(
            f"{mode} Swift SDK input has an unexpected platform matrix: "
            f"{sorted(entries)!r}; expected {sorted(expected)!r}"
        )

    for key, library in entries.items():
        declared = declared_architectures(library, key)
        if expected is not None and declared != expected[key]:
            fail(
                f"{mode} Swift SDK slice {key!r} has an unexpected architecture "
                f"contract: {sorted(declared)!r}; expected {sorted(expected[key])!r}"
            )
        framework = framework_path(xcframework, library)
        if key == ("macos", ""):
            verify_macos_layout(framework)
        binary_architectures = lipo_architectures(framework_binary(framework))
        if binary_architectures != declared:
            fail(
                f"XCFramework slice {key!r} lipo architectures "
                f"{sorted(binary_architectures)!r} do not match "
                f"SupportedArchitectures {sorted(declared)!r}"
            )

    print(
        f"verified {len(entries)} XCFramework slice(s)"
        + (f" for {mode} mode" if mode else "")
    )


def main() -> int:
    args = parse_args()
    try:
        verify_xcframework(args.xcframework, args.mode)
    except (OSError, ValueError, plistlib.InvalidFileException) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
