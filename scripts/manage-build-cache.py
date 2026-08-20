#!/usr/bin/env python3
"""Measure and safely prune repository-local Cargo build artifacts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, BinaryIO, Iterable, Iterator


DEFAULT_MAX_BYTES = 80 * 1024**3
DEFAULT_MAX_AGE_DAYS = 14
SIZE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?$", re.I)
UNIT_BYTES = {
    "": 1, "b": 1, "kb": 1000, "kib": 1024, "mb": 1000**2,
    "mib": 1024**2, "gb": 1000**3, "gib": 1024**3,
    "tb": 1000**4, "tib": 1024**4,
}


class CacheError(RuntimeError):
    """Raised when cache inspection or pruning cannot proceed safely."""


def parse_size(value: str) -> int:
    value = value.removeprefix("max_size=")
    match = SIZE_PATTERN.fullmatch(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")
    return int(float(match.group(1)) * UNIT_BYTES[(match.group(2) or "").lower()])


def parse_age(value: str) -> int:
    try:
        return int(value.removeprefix("max_age="))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid age in days: {value}") from error


def human_size(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def tree_metrics(path: Path) -> tuple[int, float]:
    if not path.exists():
        return 0, 0.0
    if path.is_file() or path.is_symlink():
        stat = path.lstat()
        return stat.st_size, stat.st_mtime
    total = 0
    newest = path.stat().st_mtime
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directories:
            candidate = root_path / name
            if candidate.is_symlink():
                stat = candidate.lstat()
                total += stat.st_size
                newest = max(newest, stat.st_mtime)
        for name in files:
            stat = (root_path / name).lstat()
            total += stat.st_size
            newest = max(newest, stat.st_mtime)
    return total, newest


def immediate_entries(path: Path) -> list[dict[str, Any]]:
    entries = []
    if path.is_dir():
        for child in path.iterdir():
            size, newest = tree_metrics(child)
            entries.append({"path": str(child), "bytes": size, "newest_mtime": newest})
    return sorted(entries, key=lambda entry: entry["bytes"], reverse=True)


def cargo_metadata(workspace: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["just", "cache-cargo-metadata"],
        cwd=workspace, check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise CacheError("cargo metadata failed; refusing build-cache management")
    return json.loads(result.stdout)


def cargo_packages(workspace: Path) -> list[str]:
    return sorted({package["name"] for package in cargo_metadata(workspace)["packages"]})


def reject_separate_build_directory(workspace: Path, managed_target: Path) -> None:
    managed_target = managed_target.resolve()
    if os.environ.get("CARGO_BUILD_BUILD_DIR"):
        raise CacheError("CARGO_BUILD_BUILD_DIR is unsupported by build-cache management")
    if not (workspace / "Cargo.toml").is_file():
        return
    metadata = cargo_metadata(workspace)
    target_directory = Path(metadata["target_directory"]).resolve()
    build_directory = Path(metadata.get("build_directory") or target_directory).resolve()
    if build_directory != target_directory:
        raise CacheError(
            "Cargo build.build-dir outside target-dir is unsupported by build-cache management: "
            f"{build_directory}"
        )
    if managed_target != target_directory:
        raise CacheError(
            "managed target directory does not match Cargo's effective target directory: "
            f"{target_directory}"
        )


def artifact_roots(target: Path, leaf: str) -> list[Path]:
    """Return host and cross-target Cargo profile artifact roots."""
    roots = [*target.glob(f"*/{leaf}"), *target.glob(f"*/*/{leaf}")]
    return sorted(path for path in roots if path.is_dir() and not path.is_symlink())


def package_metrics(target: Path, packages: Iterable[str]) -> list[dict[str, Any]]:
    normalized = {package: package.replace("-", "_") for package in packages}
    totals = {package: [0, 0.0] for package in normalized}
    roots = [*artifact_roots(target, "deps"), *artifact_roots(target, "build")]
    for root in roots:
        stems = normalized if root.name == "deps" else {package: package for package in normalized}
        for child in root.iterdir():
            for package, stem in stems.items():
                name = child.name
                if name == stem or name.startswith(f"{stem}-") or name.startswith(f"lib{stem}-"):
                    size, newest = tree_metrics(child)
                    totals[package][0] += size
                    totals[package][1] = max(totals[package][1], newest)
                    break
    return sorted(
        ({"package": package, "bytes": values[0], "newest_mtime": values[1]}
         for package, values in totals.items() if values[0]),
        key=lambda item: (item["newest_mtime"], -item["bytes"]),
    )


def active_compilers() -> list[str]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,comm=,args="], check=True, capture_output=True, text=True,
    )
    active = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) >= 2 and int(fields[0]) != os.getpid():
            if Path(fields[1]).name in {"cargo", "rustc", "rustdoc", "clippy-driver"}:
                active.append(line.strip())
    return active


@contextmanager
def cache_lock(target: Path, *, exclusive: bool, nonblocking: bool) -> Iterator[BinaryIO]:
    target.mkdir(parents=True, exist_ok=True)
    lock_file = (target / ".mesh-llm-cache-prune.lock").open("a+b")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if nonblocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(lock_file, operation)
    except BlockingIOError as error:
        lock_file.close()
        raise CacheError("build-cache cleanup or a local build is already running") from error
    try:
        yield lock_file
    finally:
        lock_file.close()


def remove_tree(path: Path, target: Path) -> None:
    candidate = Path(os.path.abspath(path))
    target_absolute = Path(os.path.abspath(target))
    target_resolved = target.resolve()
    if candidate == target_absolute or target_absolute not in candidate.parents:
        raise CacheError(f"refusing to remove path outside target: {path}")
    if path.is_symlink():
        path.unlink()
        return
    parent_resolved = path.parent.resolve()
    if parent_resolved != target_resolved and target_resolved not in parent_resolved.parents:
        raise CacheError(f"refusing to remove path through a parent outside target: {path}")
    if not path.is_dir():
        raise CacheError(f"refusing to remove non-directory path: {path}")
    shutil.rmtree(path)


def prune_incremental(
    target: Path, cutoff: float, current_bytes: int, max_bytes: int, execute: bool,
) -> tuple[int, list[dict[str, Any]]]:
    candidates = []
    for root in artifact_roots(target, "incremental"):
        for child in root.iterdir():
            size, newest = tree_metrics(child)
            if newest < cutoff or current_bytes > max_bytes:
                candidates.append((newest, child, size))
    actions = []
    for newest, path, size in sorted(candidates):
        if newest >= cutoff and current_bytes <= max_bytes:
            break
        actions.append({"kind": "incremental", "path": str(path), "bytes": size})
        if execute:
            remove_tree(path, target)
        current_bytes = max(0, current_bytes - size)
    return current_bytes, actions


def prune_packages(
    workspace: Path, target: Path, current_bytes: int, max_bytes: int,
    cutoff: float, execute: bool,
) -> tuple[int, list[dict[str, Any]]]:
    actions = []
    for metrics in package_metrics(target, cargo_packages(workspace)):
        if current_bytes <= max_bytes and metrics["newest_mtime"] >= cutoff:
            continue
        actions.append({
            "kind": "cargo-package", "package": metrics["package"],
            "estimated_bytes": metrics["bytes"],
        })
        if execute:
            environment = os.environ.copy()
            environment.update({
                "MESH_LLM_CACHE_TARGET_DIR": str(target),
                "MESH_LLM_CACHE_PACKAGE": metrics["package"],
            })
            result = subprocess.run(
                ["just", "cache-cargo-clean"],
                cwd=workspace, env=environment, check=False,
            )
            if result.returncode != 0:
                raise CacheError(f"cargo clean failed for {metrics['package']}")
            current_bytes, _ = tree_metrics(target)
        else:
            current_bytes = max(0, current_bytes - metrics["bytes"])
        if current_bytes <= max_bytes and metrics["newest_mtime"] >= cutoff:
            break
    return current_bytes, actions


def snapshot(workspace: Path, target: Path, max_bytes: int, max_age_days: int) -> dict[str, Any]:
    total, newest = tree_metrics(target)
    return {
        "schema": "mesh-llm.local-build-cache", "schema_version": 1,
        "workspace": str(workspace), "target": str(target), "target_bytes": total,
        "target_limit_bytes": max_bytes, "target_over_limit_bytes": max(0, total - max_bytes),
        "max_age_days": max_age_days, "newest_mtime": newest,
        "entries": immediate_entries(target),
    }


def render_status(report: dict[str, Any]) -> None:
    print(f"Cargo target: {human_size(report['target_bytes'])}")
    print(f"Configured limit: {human_size(report['target_limit_bytes'])}")
    print(f"Configured maximum age: {report['max_age_days']} days")
    if report["target_over_limit_bytes"]:
        print(f"Over limit: {human_size(report['target_over_limit_bytes'])}")
    print("Largest target entries:")
    for entry in report["entries"][:10]:
        print(f"  {human_size(entry['bytes']):>10}  {entry['path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "prune"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--workspace", type=Path, default=Path.cwd())
        subparser.add_argument("--target-dir", type=Path)
        subparser.add_argument("--max-size", type=parse_size, default=DEFAULT_MAX_BYTES)
        subparser.add_argument("--max-age", type=parse_age, default=DEFAULT_MAX_AGE_DAYS)
        subparser.add_argument("--json", action="store_true")
        if command == "prune":
            subparser.add_argument("--execute", action="store_true")
    build = commands.add_parser("build")
    build.add_argument("--workspace", type=Path, default=Path.cwd())
    build.add_argument("--target-dir", type=Path)
    build.add_argument("build_command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    workspace = arguments.workspace.resolve()
    target = (arguments.target_dir or workspace / "target").resolve()
    if target == workspace or workspace not in target.parents:
        raise CacheError("target directory must be a child of the workspace")
    reject_separate_build_directory(workspace, target)
    if arguments.command == "build":
        build_command = arguments.build_command
        if build_command[:1] == ["--"]:
            build_command = build_command[1:]
        if not build_command:
            raise CacheError("build command is required")
        with cache_lock(target, exclusive=False, nonblocking=False):
            return subprocess.run(build_command, cwd=workspace, check=False).returncode
    if arguments.max_age < 0:
        raise CacheError("max age must be non-negative")
    if arguments.command == "status":
        with cache_lock(target, exclusive=False, nonblocking=True):
            before = snapshot(workspace, target, arguments.max_size, arguments.max_age)
            print(json.dumps(before, indent=2, sort_keys=True)) if arguments.json else render_status(before)
            return 0
    if arguments.execute:
        with cache_lock(target, exclusive=True, nonblocking=True):
            if active_compilers():
                raise CacheError("active Cargo/Rust compiler processes detected; refusing cleanup")
            return run_prune(arguments, workspace, target)
    with cache_lock(target, exclusive=False, nonblocking=True):
        return run_prune(arguments, workspace, target)


def run_prune(arguments: argparse.Namespace, workspace: Path, target: Path) -> int:
    before = snapshot(workspace, target, arguments.max_size, arguments.max_age)
    cutoff = time.time() - arguments.max_age * 86400
    current, incremental = prune_incremental(
        target, cutoff, before["target_bytes"], arguments.max_size, arguments.execute,
    )
    current, packages = prune_packages(
        workspace, target, current, arguments.max_size, cutoff, arguments.execute,
    )
    after = snapshot(workspace, target, arguments.max_size, arguments.max_age)
    final_bytes = after["target_bytes"] if arguments.execute else current
    report = {
        "schema": "mesh-llm.local-build-cache-prune", "schema_version": 1,
        "mode": "execute" if arguments.execute else "dry-run",
        "before_bytes": before["target_bytes"], "after_bytes": final_bytes,
        "reclaimed_bytes": before["target_bytes"] - final_bytes,
        "actions": [*incremental, *packages],
    }
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Mode: {report['mode']}")
        print(f"Before: {human_size(report['before_bytes'])}")
        print(f"After: {human_size(report['after_bytes'])}")
        print(f"Reclaimed: {human_size(report['reclaimed_bytes'])}")
        for action in report["actions"]:
            identity = action.get("package", action.get("path"))
            size = action.get("estimated_bytes", action.get("bytes", 0))
            print(f"  {action['kind']}: {identity} ({human_size(size)})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CacheError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
