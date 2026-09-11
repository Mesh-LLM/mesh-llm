#!/usr/bin/env python3
"""Collect and verify redistributable Linux ELF runtime dependencies."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys


# These libraries are part of the Linux runtime and must remain host-owned.
# CUDA's driver entry point is also host-owned: the NVIDIA driver, rather than
# the CUDA toolkit, supplies libcuda.so.1 on the target machine.
HOST_LIBRARY_NAMES = {
    "ld-linux.so.2",
    "ld-linux-x86-64.so.2",
    "ld-linux-aarch64.so.1",
    "linux-vdso.so.1",
    "libc.so.6",
    "libcrypt.so.1",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libatomic.so.1",
    "libgomp.so.1",
    "libm.so.6",
    "libnsl.so.1",
    "libnuma.so.1",
    "libpthread.so.0",
    "libresolv.so.2",
    "librt.so.1",
    "libstdc++.so.6",
    "libutil.so.1",
    "libz.so.1",
    "libcuda.so",
    "libcuda.so.1",
    "libnvidia-ml.so.1",
}

# CUDA 12.9 and 13.1 list these dynamically linked Linux components as
# redistributable. Keep this list deliberately narrow: a new dependency must be
# reviewed against the matching archived EULA before it can enter a release:
# https://docs.nvidia.com/cuda/archive/12.9.1/eula/index.html
# https://docs.nvidia.com/cuda/archive/13.1.1/eula/index.html
CUDA_REDISTRIBUTABLES = (
    "libcudart",
    "libcublas",
    "libcublasLt",
    "libnvJitLink",
)


class ElfFormatError(ValueError):
    """Raised when an ELF file is malformed or cannot be inspected."""


class ElfImage:
    def __init__(
        self,
        path: pathlib.Path,
        needed: tuple[str, ...],
        soname: str | None,
        elf_class: str,
        machine: str,
    ) -> None:
        self.path = path
        self.needed = needed
        self.soname = soname
        self.elf_class = elf_class
        self.machine = machine


def _readelf(*arguments: str, path: pathlib.Path) -> str:
    try:
        result = subprocess.run(
            ["readelf", *arguments, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("readelf is required to inspect Linux native runtime ELF files") from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise ElfFormatError(f"cannot inspect ELF file {path}: {details}") from error
    return result.stdout


def _is_elf(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError as error:
        raise RuntimeError(f"read ELF file {path}: {error}") from error


def _dynamic_entries(path: pathlib.Path) -> tuple[list[str], str | None]:
    output = _readelf("-d", path=path)
    needed: list[str] = []
    soname: str | None = None
    dynamic_re = re.compile(r"\((NEEDED|SONAME)\).*\[(.*)\]")
    for line in output.splitlines():
        match = dynamic_re.search(line)
        if not match:
            continue
        tag, value = match.groups()
        if tag == "NEEDED":
            needed.append(value)
        else:
            soname = value
    return needed, soname


def _elf_image(path: pathlib.Path) -> ElfImage | None:
    if not _is_elf(path):
        return None
    header = _readelf("-h", path=path)
    elf_class = ""
    machine = ""
    for line in header.splitlines():
        if line.startswith("  Class:"):
            elf_class = line.split(":", 1)[1].strip()
        elif line.startswith("  Machine:"):
            machine = line.split(":", 1)[1].strip()
    if not elf_class or not machine:
        raise ElfFormatError(f"ELF header is missing class or machine: {path}")
    needed, soname = _dynamic_entries(path)
    return ElfImage(path, tuple(needed), soname, elf_class, machine)


def _iter_files(directories: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        try:
            entries = sorted(directory.rglob("*"))
        except OSError as error:
            raise RuntimeError(f"scan Linux runtime directory {directory}: {error}") from error
        for entry in entries:
            if not entry.is_file() or entry.is_symlink():
                continue
            resolved = entry.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(entry)
    return sorted(files, key=lambda value: value.as_posix())


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_stub(path: pathlib.Path) -> bool:
    return any(part.casefold() == "stubs" for part in path.parts)


def _architecture_matches(image: ElfImage, arch: str | None) -> bool:
    if arch is None:
        return True
    expected = {
        "x86_64": ("ELF64", "Advanced Micro Devices X86-64"),
        "aarch64": ("ELF64", "AArch64"),
        "arm": ("ELF32", "ARM"),
    }.get(arch)
    if expected is None:
        raise RuntimeError(f"unsupported Linux runtime architecture: {arch}")
    expected_class, expected_machine = expected
    return image.elf_class == expected_class and image.machine == expected_machine


def _check_arch(image: ElfImage, arch: str | None, label: str) -> None:
    if not _architecture_matches(image, arch):
        raise RuntimeError(
            f"wrong architecture for {label}: {image.path} is "
            f"{image.elf_class}/{image.machine}, expected {arch}"
        )


def _aliases(image: ElfImage) -> set[str]:
    result = {image.path.name}
    if image.soname:
        result.add(image.soname)
    return result


def _index_images(
    paths: list[pathlib.Path], *, arch: str | None, label: str, check_arch: bool = True
) -> tuple[list[ElfImage], dict[str, ElfImage]]:
    images: list[ElfImage] = []
    aliases: dict[str, ElfImage] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        image = _elf_image(path)
        if image is None:
            continue
        if check_arch:
            _check_arch(image, arch, label)
        images.append(image)
        digest = _sha256(path)
        for alias in _aliases(image):
            previous = aliases.get(alias)
            if previous is not None:
                previous_digest = hashes[alias]
                if previous_digest != digest:
                    raise RuntimeError(
                        f"conflicting ELF libraries provide {alias}: "
                        f"{previous.path} and {path}"
                    )
                continue
            aliases[alias] = image
            hashes[alias] = digest
    return images, aliases


def _host_owned(name: str) -> bool:
    lower = name.casefold()
    return (
        lower in HOST_LIBRARY_NAMES
        or lower.startswith("linux-vdso")
        or lower.startswith("libnvidia-")
    )


def _package_index(lib_dir: pathlib.Path, scan_dirs: list[pathlib.Path], arch: str | None):
    paths = _iter_files([lib_dir, *scan_dirs])
    images, aliases = _index_images(paths, arch=arch, label="packaged runtime")
    names = {image.path.name: image for image in images if image.path.parent == lib_dir}
    return images, names, aliases


def dependency_gaps(
    lib_dir: pathlib.Path,
    scan_dirs: list[pathlib.Path] | None = None,
    *,
    arch: str | None = None,
) -> dict[str, set[str]]:
    scan_dirs = scan_dirs or []
    images, _, _ = _package_index(lib_dir, scan_dirs, arch)
    packaged_host_owned = sorted(
        image.path.name
        for image in images
        if image.path.parent == lib_dir and _host_owned(image.path.name)
    )
    if packaged_host_owned:
        raise RuntimeError(
            "host-owned Linux libraries must not be packaged: "
            + ", ".join(packaged_host_owned)
        )
    packaged_names = {image.path.name for image in images}
    gaps: dict[str, set[str]] = {}
    for image in images:
        missing = {
            dependency
            for dependency in image.needed
            if not _host_owned(dependency) and dependency not in packaged_names
        }
        if missing:
            gaps[image.path.name] = missing
    return gaps


def _candidate_index(paths: list[pathlib.Path]) -> dict[str, list[ElfImage]]:
    result: dict[str, list[ElfImage]] = {}
    for path in paths:
        image = _elf_image(path)
        if image is None:
            continue
        for alias in _aliases(image):
            result.setdefault(alias, []).append(image)
    return result


def _search_index(
    search_dirs: list[pathlib.Path],
) -> tuple[dict[str, list[ElfImage]], dict[str, list[ElfImage]]]:
    paths = _iter_files(search_dirs)
    real_paths = [path for path in paths if not _is_stub(path)]
    stub_paths = [path for path in paths if _is_stub(path)]
    return _candidate_index(real_paths), _candidate_index(stub_paths)


def _select_provider(
    dependency: str,
    candidates: list[ElfImage],
    *,
    arch: str | None,
) -> ElfImage:
    matching = [image for image in candidates if _architecture_matches(image, arch)]
    if not matching:
        available = ", ".join(
            f"{image.path} ({image.elf_class}/{image.machine})" for image in candidates
        )
        raise RuntimeError(
            f"wrong architecture for search dependency {dependency}; "
            f"expected {arch}, found: {available}"
        )
    by_digest: dict[str, ElfImage] = {}
    for image in matching:
        by_digest.setdefault(_sha256(image.path), image)
    if len(by_digest) > 1:
        paths = ", ".join(str(image.path) for image in by_digest.values())
        raise RuntimeError(f"conflicting ELF libraries provide {dependency}: {paths}")
    return next(iter(by_digest.values()))


def _validate_cuda_redistributable(dependency: str, cuda_major: int) -> None:
    match = re.fullmatch(
        rf"({'|'.join(re.escape(name) for name in CUDA_REDISTRIBUTABLES)})"
        rf"\.so\.{cuda_major}(?:\..+)?",
        dependency,
    )
    if match is None:
        raise RuntimeError(
            "Linux CUDA runtime dependency is not in the reviewed "
            f"redistributable allowlist for CUDA {cuda_major}: {dependency}"
        )


def _copy_dependency(
    source: ElfImage,
    dependency: str,
    lib_dir: pathlib.Path,
    *,
    arch: str | None,
) -> pathlib.Path:
    _check_arch(source, arch, "search dependency")
    destination = lib_dir / dependency
    if destination.exists():
        if _sha256(destination) != _sha256(source.path):
            raise RuntimeError(
                f"conflicting packaged dependency {dependency}: "
                f"{destination} and {source.path}"
            )
        return destination
    shutil.copy2(source.path, destination)
    return destination


def collect_dependencies(
    lib_dir: pathlib.Path,
    search_dirs: list[pathlib.Path],
    scan_dirs: list[pathlib.Path] | None = None,
    *,
    arch: str | None = None,
    cuda_major: int,
) -> list[pathlib.Path]:
    scan_dirs = scan_dirs or []
    search_index, stub_index = _search_index(search_dirs)
    copied: list[pathlib.Path] = []
    while True:
        gaps = dependency_gaps(lib_dir, scan_dirs, arch=arch)
        if not gaps:
            return copied
        unresolved: dict[str, set[str]] = {}
        for importer, dependencies in sorted(gaps.items()):
            for dependency in sorted(dependencies):
                candidates = search_index.get(dependency, [])
                if not candidates:
                    if dependency in stub_index:
                        stub = stub_index[dependency][0]
                        raise RuntimeError(
                            f"CUDA stub library cannot satisfy runtime dependency "
                            f"{dependency}: {stub.path}"
                        )
                    unresolved.setdefault(importer, set()).add(dependency)
                    continue
                _validate_cuda_redistributable(dependency, cuda_major)
                source = _select_provider(dependency, candidates, arch=arch)
                destination = _copy_dependency(source, dependency, lib_dir, arch=arch)
                if destination not in copied:
                    copied.append(destination)
        if unresolved:
            details = "; ".join(
                f"{importer}: {', '.join(sorted(dependencies))}"
                for importer, dependencies in sorted(unresolved.items())
            )
            raise RuntimeError(f"unresolved Linux runtime ELF dependencies: {details}")


def verify_dependencies(
    lib_dir: pathlib.Path,
    scan_dirs: list[pathlib.Path] | None = None,
    *,
    arch: str | None = None,
) -> None:
    gaps = dependency_gaps(lib_dir, scan_dirs, arch=arch)
    if not gaps:
        return
    details = "; ".join(
        f"{importer}: {', '.join(sorted(dependencies))}"
        for importer, dependencies in sorted(gaps.items())
    )
    raise RuntimeError(f"unpackaged Linux runtime ELF dependencies: {details}")


def dependency_order(
    lib_dir: pathlib.Path,
    scan_dirs: list[pathlib.Path] | None = None,
    *,
    primary: str,
    arch: str | None = None,
) -> list[pathlib.Path]:
    scan_dirs = scan_dirs or []
    verify_dependencies(lib_dir, scan_dirs, arch=arch)
    images, _, _ = _package_index(lib_dir, scan_dirs, arch)
    libraries = {
        image.path.name: image
        for image in images
        if image.path.parent == lib_dir
    }
    all_library_paths = sorted(
        path for path in lib_dir.iterdir() if path.is_file() and not path.is_symlink()
    )
    if not any(path.name == primary for path in all_library_paths):
        raise RuntimeError(f"primary Linux runtime library is missing: {primary}")
    # Lightweight packaging fixtures and downstream test doubles may use a
    # non-ELF placeholder. There is no dependency graph to order in that case,
    # but the primary library still belongs last in the manifest.
    if not libraries:
        return [path for path in all_library_paths if path.name != primary] + [
            lib_dir / primary
        ]
    ordered: list[pathlib.Path] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise RuntimeError(f"cyclic Linux runtime dependency graph at {name}")
        image = libraries.get(name)
        if image is None:
            return
        visiting.add(name)
        for dependency in sorted(image.needed):
            if dependency in libraries:
                visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(image.path)

    for name in sorted(libraries):
        if name != primary:
            visit(name)
    if primary in libraries:
        visit(primary)
        primary_path = ordered.pop()
    else:
        primary_path = lib_dir / primary
    ordered_names = {path.name for path in ordered}
    ordered.extend(
        path
        for path in all_library_paths
        if path.name not in ordered_names and path.name != primary
    )
    ordered.append(primary_path)
    if len(ordered) != len(all_library_paths):
        raise RuntimeError("could not order all Linux runtime libraries")
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("collect", "verify", "order"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--lib-dir", type=pathlib.Path, required=True)
        subparser.add_argument("--scan-dir", type=pathlib.Path, action="append", default=[])
        subparser.add_argument("--arch", choices=("x86_64", "aarch64", "arm"))
        if command == "collect":
            subparser.add_argument("--search-dir", type=pathlib.Path, action="append", default=[])
            subparser.add_argument("--cuda-major", type=int, choices=(12, 13), required=True)
        if command == "order":
            subparser.add_argument("--primary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "collect":
            copied = collect_dependencies(
                args.lib_dir,
                args.search_dir,
                [args.lib_dir, *args.scan_dir],
                arch=args.arch,
                cuda_major=args.cuda_major,
            )
            for path in copied:
                print(f"bundled Linux runtime dependency: {path.name}")
        elif args.command == "verify":
            verify_dependencies(args.lib_dir, [args.lib_dir, *args.scan_dir], arch=args.arch)
        else:
            for path in dependency_order(
                args.lib_dir,
                [args.lib_dir, *args.scan_dir],
                primary=args.primary,
                arch=args.arch,
            ):
                print(path.name)
    except (OSError, ElfFormatError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
