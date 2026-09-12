#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


FORBIDDEN_IMPORTS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|[/\\])libcuda(?:\.so|\.dylib|\.dll|$)",
        r"(^|[/\\])libcudart",
        r"(^|[/\\])libcublas",
        r"(^|[/\\])libnccl",
        r"(^|[/\\])libamdhip64",
        r"(^|[/\\])libhip",
        r"(^|[/\\])libhsa-runtime",
        r"(^|[/\\])nvcuda\.dll$",
        r"(^|[/\\])cudart64_[^/\\]+\.dll$",
        r"(^|[/\\])cublas(?:lt)?64_[^/\\]+\.dll$",
        r"(^|[/\\])amdhip64\.dll$",
        r"(^|[/\\])hipblas\.dll$",
        r"(^|[/\\])rocblas\.dll$",
        r"(^|[/\\])libvulkan",
        r"(^|[/\\])vulkan-1\.dll$",
        r"Metal\.framework",
        r"(^|[/\\])(?:lib)?ggml",
        r"(^|[/\\])(?:lib)?llama",
    )
)


GLIBC_VERSION_RE = re.compile(r"GLIBC_(\d+)\.(\d+)")
VERSION_NEEDS_HEADING = "Version needs section"


def parse_version(value: str) -> tuple[int, int]:
    major, _, minor = value.strip().partition(".")
    return int(major), int(minor)


def format_version(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def read_declared_glibc_floor(path: Path) -> tuple[int, int]:
    """Reads the shared floor file, ignoring its explanatory comments."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return parse_version(stripped)
    raise ValueError(f"no glibc floor declared in {path}")


def parse_elf_glibc_floor(output: str) -> tuple[int, int] | None:
    """Highest GLIBC symbol version the binary needs from the host's libc.

    Only the version-needs section counts. A shared library's own version
    definitions live in the same `readelf -V` output and say nothing about
    what libc it will refuse to load against.
    """
    _, heading, needs = output.partition(VERSION_NEEDS_HEADING)
    if not heading:
        return None
    versions = [
        (int(major), int(minor))
        for major, minor in GLIBC_VERSION_RE.findall(needs)
    ]
    return max(versions) if versions else None


def inspect_glibc_floor(path: Path) -> tuple[int, int] | None:
    return parse_elf_glibc_floor(run_tool(("readelf", "-V", str(path))))


def binary_format(path: Path) -> str:
    header = path.read_bytes()[:4]
    if header == b"\x7fELF":
        return "elf"
    if header[:2] == b"MZ":
        return "pe"
    if header in (
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    ):
        return "macho"
    raise ValueError(f"unsupported host executable format: {path}")


def parse_elf_imports(output: str) -> list[str]:
    return sorted(set(re.findall(r"\(NEEDED\).*\[([^\]]+)\]", output)))


def parse_macho_imports(output: str) -> list[str]:
    imports = []
    for line in output.splitlines():
        if not line[:1].isspace():
            continue
        value = line.strip().split(" (compatibility version", 1)[0]
        if value:
            imports.append(value)
    return sorted(set(imports))


def parse_pe_imports(output: str) -> list[str]:
    imports = []
    for line in output.splitlines():
        match = re.search(r"(?:DLL Name:|Name:)\s*(\S+\.dll)\b", line, re.IGNORECASE)
        if match:
            imports.append(match.group(1))
    return sorted(set(imports))


def inspect_dependencies(path: Path, format_name: str | None = None) -> tuple[str, list[str]]:
    format_name = format_name or binary_format(path)
    if format_name == "elf":
        output = run_tool(("readelf", "-d", str(path)))
        imports = parse_elf_imports(output)
    elif format_name == "macho":
        output = run_tool(("otool", "-L", str(path)))
        imports = parse_macho_imports(output)
    elif format_name == "pe":
        if shutil.which("llvm-readobj"):
            output = run_tool(("llvm-readobj", "--coff-imports", str(path)))
        else:
            output = run_tool(("objdump", "-p", str(path)))
        imports = parse_pe_imports(output)
    else:
        raise ValueError(f"unsupported host executable format: {format_name}")
    return format_name, imports


def run_tool(command: tuple[str, ...]) -> str:
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"{command[0]} is required to inspect host dependencies")
    # parse_elf_glibc_floor looks for the English "Version needs section"
    # heading; a localized readelf would make the glibc floor check pass
    # silently, so pin the C locale.
    return subprocess.check_output(
        command,
        text=True,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LC_ALL": "C"},
    )


def forbidden_imports(imports: list[str]) -> list[str]:
    return [
        dependency
        for dependency in imports
        if any(pattern.search(dependency) for pattern in FORBIDDEN_IMPORTS)
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--format", choices=("elf", "macho", "pe"))
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--no-import-policy",
        action="store_true",
        help=(
            "Report imports without enforcing the host policy. Native runtime "
            "libraries legitimately import each other, so only the glibc floor "
            "applies to them."
        ),
    )
    parser.add_argument(
        "--max-glibc",
        help=(
            "Reject an ELF binary that needs a GLIBC symbol version above this "
            "one, or the literal 'declared' to read scripts/linux-glibc-floor.txt."
        ),
    )
    return parser.parse_args(argv)


def resolve_max_glibc(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if value == "declared":
        return read_declared_glibc_floor(
            Path(__file__).resolve().parent / "linux-glibc-floor.txt"
        )
    return parse_version(value)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        format_name, imports = inspect_dependencies(args.binary, args.format)
        rejected = [] if args.no_import_policy else forbidden_imports(imports)
        max_glibc = resolve_max_glibc(args.max_glibc)
        glibc_floor = inspect_glibc_floor(args.binary) if format_name == "elf" else None
        report = {
            "binary": args.binary.name,
            "format": format_name,
            "glibc_floor": format_version(glibc_floor) if glibc_floor else None,
            "imports": imports,
            "policy": "none" if args.no_import_policy else "mesh-llm-dynamic-host-v2",
            "rejected_imports": rejected,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, sort_keys=True))
        if rejected:
            print(
                "host dependency policy rejected: " + ", ".join(rejected),
                file=sys.stderr,
            )
            return 1
        if max_glibc and glibc_floor and glibc_floor > max_glibc:
            print(
                f"{args.binary.name} needs GLIBC_{format_version(glibc_floor)} but the "
                f"declared floor is {format_version(max_glibc)}. Raising the floor drops "
                "Linux distributions that were supported before; see mesh-llm#1522.",
                file=sys.stderr,
            )
            return 1
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
