#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


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
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        format_name, imports = inspect_dependencies(args.binary, args.format)
        rejected = forbidden_imports(imports)
        report = {
            "binary": args.binary.name,
            "format": format_name,
            "imports": imports,
            "policy": "mesh-llm-dynamic-host-v2",
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
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
