#!/usr/bin/env python3
"""Collect and verify non-system DLL dependencies in Windows native runtimes."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import struct
import sys


HOST_DLLS = {
    "advapi32.dll",
    "bcrypt.dll",
    "cfgmgr32.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "d3d12.dll",
    "dbghelp.dll",
    "dxgi.dll",
    "gdi32.dll",
    "kernel32.dll",
    "msvcp140.dll",
    "msvcrt.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "psapi.dll",
    "rpcrt4.dll",
    "secur32.dll",
    "setupapi.dll",
    "shell32.dll",
    "shlwapi.dll",
    "ucrtbase.dll",
    "user32.dll",
    "version.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vulkan-1.dll",
    "winmm.dll",
    "ws2_32.dll",
}


class PeFormatError(ValueError):
    """Raised when a file is not a supported PE image."""


def _unpack(data: bytes, fmt: str, offset: int) -> tuple[int, ...]:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise PeFormatError("truncated PE image")
    return struct.unpack_from(fmt, data, offset)


def _cstring(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        raise PeFormatError("PE import name points outside the image")
    end = data.find(b"\0", offset)
    if end < 0:
        raise PeFormatError("unterminated PE import name")
    return data[offset:end].decode("ascii")


def imported_dlls(path: pathlib.Path) -> list[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise PeFormatError(f"not a PE image: {path}")
    (pe_offset,) = _unpack(data, "<I", 0x3C)
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise PeFormatError(f"invalid PE signature: {path}")

    file_header = pe_offset + 4
    (_, section_count, _, _, _, optional_size, _) = _unpack(
        data, "<HHIIIHH", file_header
    )
    optional_header = file_header + 20
    (magic,) = _unpack(data, "<H", optional_header)
    if magic == 0x20B:
        data_directories = optional_header + 112
    elif magic == 0x10B:
        data_directories = optional_header + 96
    else:
        raise PeFormatError(f"unsupported PE optional-header magic 0x{magic:x}: {path}")

    (import_rva, _) = _unpack(data, "<II", data_directories + 8)
    section_table = optional_header + optional_size
    sections: list[tuple[int, int, int]] = []
    for index in range(section_count):
        section = section_table + index * 40
        (virtual_size, virtual_address, raw_size, raw_offset) = _unpack(
            data, "<IIII", section + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def rva_offset(rva: int) -> int:
        for virtual_address, size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_offset + rva - virtual_address
        if rva < len(data):
            return rva
        raise PeFormatError(f"PE RVA 0x{rva:x} is outside every section: {path}")

    if import_rva == 0:
        return []
    descriptor = rva_offset(import_rva)
    imports: list[str] = []
    while True:
        fields = _unpack(data, "<IIIII", descriptor)
        if not any(fields):
            break
        name_rva = fields[3]
        if name_rva == 0:
            raise PeFormatError(f"PE import descriptor has no DLL name: {path}")
        imports.append(_cstring(data, rva_offset(name_rva)))
        descriptor += 20
    return imports


def is_host_dll(name: str) -> bool:
    normalized = name.casefold()
    return (
        normalized in HOST_DLLS
        or normalized.startswith("api-ms-win-")
        or normalized.startswith("ext-ms-win-")
    )


def default_search_dirs() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        candidates.extend(
            [pathlib.Path(vulkan_sdk) / "Bin", pathlib.Path(vulkan_sdk) / "Bin32"]
        )
    candidates.extend(
        pathlib.Path(entry)
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
    )
    return candidates


def _dll_index(directories: list[pathlib.Path]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        try:
            entries = directory.iterdir()
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and entry.suffix.casefold() == ".dll":
                result.setdefault(entry.name.casefold(), entry)
    return result


def _packaged_dlls(lib_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        path.name.casefold(): path
        for path in lib_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".dll"
    }


def dependency_gaps(lib_dir: pathlib.Path) -> dict[str, set[str]]:
    packaged = _packaged_dlls(lib_dir)
    gaps: dict[str, set[str]] = {}
    for name, library in sorted(packaged.items()):
        missing = {
            dependency
            for dependency in imported_dlls(library)
            if not is_host_dll(dependency) and dependency.casefold() not in packaged
        }
        if missing:
            gaps[name] = missing
    return gaps


def collect_dependencies(
    lib_dir: pathlib.Path, search_dirs: list[pathlib.Path]
) -> list[pathlib.Path]:
    search_index = _dll_index([lib_dir, *search_dirs, *default_search_dirs()])
    copied: list[pathlib.Path] = []
    while True:
        gaps = dependency_gaps(lib_dir)
        if not gaps:
            return copied
        unresolved: dict[str, set[str]] = {}
        for importer, dependencies in gaps.items():
            for dependency in sorted(dependencies):
                source = search_index.get(dependency.casefold())
                if source is None:
                    unresolved.setdefault(importer, set()).add(dependency)
                    continue
                destination = lib_dir / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
                    copied.append(destination)
        if unresolved:
            details = "; ".join(
                f"{importer}: {', '.join(sorted(dependencies))}"
                for importer, dependencies in sorted(unresolved.items())
            )
            raise RuntimeError(f"unresolved Windows runtime DLL dependencies: {details}")


def verify_dependencies(lib_dir: pathlib.Path) -> None:
    gaps = dependency_gaps(lib_dir)
    if not gaps:
        return
    details = "; ".join(
        f"{importer}: {', '.join(sorted(dependencies))}"
        for importer, dependencies in sorted(gaps.items())
    )
    raise RuntimeError(f"unpackaged Windows runtime DLL dependencies: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--lib-dir", type=pathlib.Path, required=True)
    collect.add_argument("--search-dir", type=pathlib.Path, action="append", default=[])
    verify = subparsers.add_parser("verify")
    verify.add_argument("--lib-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "collect":
            copied = collect_dependencies(args.lib_dir, args.search_dir)
            for path in copied:
                print(f"bundled Windows runtime dependency: {path.name}")
        else:
            verify_dependencies(args.lib_dir)
    except (OSError, PeFormatError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
