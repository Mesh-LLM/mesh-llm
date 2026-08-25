#!/usr/bin/env python3
"""Extract a tar archive without permitting writes outside the destination."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def normalized_parts(
    raw_name: str,
    *,
    label: str,
    allow_root: bool = False,
) -> tuple[str, ...]:
    """Execute normalized parts operation."""
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ValueError(f"unsafe {label}: {raw_name!r}")
    if raw_name.startswith("/") or WINDOWS_DRIVE.match(raw_name):
        raise ValueError(f"absolute {label} is not allowed: {raw_name}")
    parts = tuple(part for part in PurePosixPath(raw_name).parts if part != ".")
    if not parts:
        if allow_root:
            return ()
        raise ValueError(f"empty {label} is not allowed: {raw_name!r}")
    if any(part in ("", "..") for part in parts):
        raise ValueError(f"traversing {label} is not allowed: {raw_name}")
    return parts


def destination_path(root: Path, parts: tuple[str, ...]) -> Path:
    """Execute destination path operation."""
    return root.joinpath(*parts)


def validate_members(
    archive: tarfile.TarFile,
) -> list[tuple[tarfile.TarInfo, tuple[str, ...]]]:
    """Validate members.

    Raises:
        ValidationError: If validation fails.
    """
    validated: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for member in archive.getmembers():
        parts = normalized_parts(
            member.name,
            label="archive member path",
            allow_root=member.isdir(),
        )
        if parts in seen:
            raise ValueError(f"duplicate archive member path: {member.name}")
        seen.add(parts)
        if not parts:
            continue
        if not (
            member.isdir()
            or member.isreg()
            or member.issym()
            or member.islnk()
        ):
            raise ValueError(
                f"unsupported archive member type for {member.name}: "
                f"{member.type!r}"
            )
        if member.issym() or member.islnk():
            link_parts = normalized_parts(
                member.linkname,
                label=f"link target for {member.name}",
            )
            target_parts = (
                (*parts[:-1], *link_parts) if member.issym() else link_parts
            )
            normalized_target: list[str] = []
            for part in target_parts:
                if part == "..":
                    if not normalized_target:
                        raise ValueError(
                            f"archive link escapes destination: {member.name}"
                        )
                    normalized_target.pop()
                elif part != ".":
                    normalized_target.append(part)
            if not normalized_target:
                raise ValueError(
                    f"archive link has an empty target: {member.name}"
                )
        validated.append((member, parts))
    return validated


def apply_mode(path: Path, member: tarfile.TarInfo) -> None:
    """Execute apply mode operation."""
    if os.name != "nt":
        path.chmod(member.mode & 0o777)


def safe_extract(archive_path: Path, destination: Path) -> None:
    """Execute safe extract operation."""
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(
            f"extraction destination cannot be a symlink: {destination}"
        )
    root = destination.resolve(strict=True)
    if any(root.iterdir()):
        raise ValueError(f"extraction destination must be empty: {root}")
    with tarfile.open(archive_path, mode="r:*") as archive:
        validated = validate_members(archive)

        directories = [
            (member, parts)
            for member, parts in validated
            if member.isdir()
        ]
        regular_files = [
            (member, parts)
            for member, parts in validated
            if member.isreg()
        ]
        links = [
            (member, parts)
            for member, parts in validated
            if member.issym() or member.islnk()
        ]

        for member, parts in directories:
            destination_path(root, parts).mkdir(parents=True, exist_ok=True)

        for member, parts in regular_files:
            output = destination_path(root, parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() or output.is_symlink():
                raise ValueError(
                    f"archive member would overwrite an existing path: "
                    f"{member.name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(
                    f"archive member has no file payload: {member.name}"
                )
            with source, output.open("xb") as handle:
                shutil.copyfileobj(source, handle)
            apply_mode(output, member)

        for member, parts in links:
            output = destination_path(root, parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() or output.is_symlink():
                raise ValueError(
                    f"archive link would overwrite an existing path: "
                    f"{member.name}"
                )
            if member.issym():
                os.symlink(member.linkname, output)
            else:
                link_parts = normalized_parts(
                    member.linkname,
                    label=f"hard-link target for {member.name}",
                )
                target = destination_path(root, link_parts)
                if not target.is_file() or target.is_symlink():
                    raise ValueError(
                        f"hard-link target is not a regular extracted file: "
                        f"{member.linkname}"
                    )
                os.link(target, output)

        for member, parts in reversed(directories):
            apply_mode(destination_path(root, parts), member)


def main() -> int:
    """Execute main program logic.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        safe_extract(args.archive, args.destination)
    except (OSError, tarfile.TarError, ValueError) as error:
        parser.exit(1, f"unsafe or invalid tar archive: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
