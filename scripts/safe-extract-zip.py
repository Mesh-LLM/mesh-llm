#!/usr/bin/env python3
"""Extract a ZIP archive without allowing path or symlink escapes."""

from __future__ import annotations

import re
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn


WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class Entry:
    """Represents Entry functionality."""
    info: zipfile.ZipInfo
    parts: tuple[str, ...]
    kind: str
    mode: int
    link_target: str | None = None


def fail(message: str) -> NoReturn:
    """Execute fail operation."""
    raise SystemExit(f"unsafe ZIP archive: {message}")


def portable_parts(name: str, *, label: str) -> tuple[str, ...]:
    """Execute portable parts operation."""
    if (
        not name
        or any(character in name for character in ("\0", "\r", "\n", "\t"))
        or "\\" in name
        or name.startswith("/")
        or WINDOWS_DRIVE.match(name)
    ):
        fail(f"{label} is not a portable relative path: {name!r}")

    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        fail(f"{label} escapes the extraction root: {name!r}")
    return tuple(path.parts)


def resolve_link(parts: tuple[str, ...], target: str) -> None:
    """Execute resolve link operation."""
    if (
        not target
        or any(character in target for character in ("\0", "\r", "\n", "\t"))
        or "\\" in target
        or target.startswith("/")
        or WINDOWS_DRIVE.match(target)
    ):
        fail(f"symlink target is not portable: {target!r}")

    resolved = list(parts[:-1])
    for part in PurePosixPath(target).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved:
                fail(f"symlink target escapes the extraction root: {target!r}")
            resolved.pop()
            continue
        resolved.append(part)
    if not resolved:
        fail(f"symlink target resolves to the extraction root: {target!r}")


def classify(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> Entry:
    """Execute classify operation."""
    name = info.filename.rstrip("/") if info.is_dir() else info.filename
    parts = portable_parts(name, label="entry")
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)

    if info.is_dir() or file_type == stat.S_IFDIR:
        return Entry(info, parts, "directory", mode)
    if file_type == stat.S_IFLNK:
        try:
            target = archive.read(info).decode("utf-8")
        except UnicodeDecodeError:
            fail(f"symlink target is not UTF-8: {info.filename!r}")
        resolve_link(parts, target)
        return Entry(info, parts, "symlink", mode, target)
    if file_type in {0, stat.S_IFREG}:
        return Entry(info, parts, "file", mode)
    fail(f"unsupported entry type for {info.filename!r}")


def inspect_archive(archive: zipfile.ZipFile) -> list[Entry]:
    """Execute inspect archive operation."""
    entries = [classify(archive, info) for info in archive.infolist()]
    seen: set[tuple[str, ...]] = set()
    symlinks = {entry.parts for entry in entries if entry.kind == "symlink"}

    for entry in entries:
        if entry.parts in seen:
            fail(f"duplicate entry path: {entry.info.filename!r}")
        seen.add(entry.parts)
        for index in range(1, len(entry.parts)):
            if entry.parts[:index] in symlinks:
                fail(
                    "entry is nested beneath an archive symlink: "
                    f"{entry.info.filename!r}"
                )
    return entries


def extract(archive_path: Path, destination: Path) -> None:
    """Execute extract operation."""
    if not archive_path.is_file():
        fail(f"archive does not exist: {archive_path}")
    if destination.is_symlink():
        fail(f"destination cannot be a symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        fail(f"destination must be empty: {destination}")

    with zipfile.ZipFile(archive_path) as archive:
        entries = inspect_archive(archive)

        for entry in entries:
            if entry.kind == "directory":
                destination.joinpath(*entry.parts).mkdir(parents=True, exist_ok=True)

        for entry in entries:
            if entry.kind != "file":
                continue
            output = destination.joinpath(*entry.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry.info) as source, output.open("xb") as target:
                shutil.copyfileobj(source, target)
            permissions = entry.mode & 0o777
            if permissions:
                output.chmod(permissions)

        for entry in entries:
            if entry.kind != "symlink":
                continue
            output = destination.joinpath(*entry.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.symlink_to(entry.link_target)


def main() -> None:
    """Execute main program logic.

    Returns:
        Exit code.
    """
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: scripts/safe-extract-zip.py ARCHIVE.zip DESTINATION"
        )
    extract(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
