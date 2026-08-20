#!/usr/bin/env python3
"""Check the audited Rust process-environment mutation contract.

`std::env::set_var` and `remove_var` are unsafe on Rust 2024 platforms where a
concurrent environment reader can race the mutation.  The audited call sites
fall into two deliberately separate contracts:

* test-only overrides are scoped by a guard and every test that owns one is
  marked ``#[serial]``;
* build scripts run in their own process, while the four runtime calls that do
  not yet have a proven single-threaded boundary remain explicit TODOs.

The source tree contains several independent crates (and two build scripts),
so a shared Rust test helper would introduce an unnecessary dev-dependency
coupling.  This repository-level check keeps the contract visible across those
crate boundaries instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


TODO = "// TODO: Audit that the environment access only happens in single-threaded code."

# This is the complete census of the 128 TODO comments that this check was
# introduced to audit.  Keep the list explicit so a new environment mutation
# cannot silently join the audit surface without review.
AUDITED_FILES = (
    "crates/skippy-protocol/build.rs",
    "crates/mesh-llm-plugin/build.rs",
    "crates/mesh-llm-host-runtime/src/capture.rs",
    "crates/mesh-llm-host-runtime/src/runtime/instance.rs",
    "crates/mesh-llm-host-runtime/src/models/maintenance.rs",
    "crates/mesh-llm-host-runtime/src/models/remote_catalog.rs",
    "crates/mesh-llm-host-runtime/src/models/artifact_transfer.rs",
    "crates/mesh-llm-host-runtime/src/models/delete_tests.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/package_download.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/cache_management.rs",
    "crates/model-hf/src/store/local.rs",
    "crates/mesh-llm-system/src/autoupdate.rs",
    "crates/mesh-llm-system/src/autoupdate/release_fetch.rs",
    "crates/mesh-llm-system/src/benchmark/tests.rs",
    "crates/skippy-runtime/src/logging.rs",
    "crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
)

# These are the only intentionally unresolved sites.  They execute on runtime
# startup / native-runtime setup paths that may already have Tokio worker
# threads, so replacing the TODO with a guessed SAFETY claim would be unsafe.
DEFERRED_FILES = {
    "crates/skippy-protocol/build.rs",
    "crates/mesh-llm-plugin/build.rs",
    "crates/skippy-runtime/src/logging.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
}

MUTATION_RE = re.compile(r"(?:std::)?env::(?:set_var|remove_var)\s*\(")
FUNCTION_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def mutation_lines(lines: list[str]) -> list[int]:
    """Return zero-based lines containing a process-env mutation call."""

    return [index for index, line in enumerate(lines) if MUTATION_RE.search(line)]


def nearest_function(lines: list[str], line_index: int) -> tuple[int, str] | None:
    for index in range(line_index, -1, -1):
        match = FUNCTION_RE.search(lines[index])
        if match:
            return index, match.group(1)
    return None


def preceding_comments(lines: list[str], line_index: int, limit: int = 12) -> str:
    start = max(0, line_index - limit)
    return "\n".join(lines[start:line_index])


def test_contract(lines: list[str], line_index: int, function: tuple[int, str] | None) -> bool:
    """Whether a mutation has an explicit serial-test contract nearby."""

    nearby = preceding_comments(lines, line_index)
    if "#[serial]" in nearby or "serial test" in nearby.lower():
        return True
    if function is None:
        return False
    function_index, _ = function
    attrs = "\n".join(lines[max(0, function_index - 8) : function_index])
    return "#[serial]" in attrs


def check_file(root: Path, relative_path: str) -> list[str]:
    path = root / relative_path
    if not path.is_file():
        return [f"{relative_path}: audited source file is missing"]

    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    is_build_script = path.name == "build.rs"
    is_deferred = relative_path in DEFERRED_FILES
    has_test_module = (
        any("#[cfg(test)]" in line for line in lines)
        or "/tests/" in relative_path
        or path.name == "tests.rs"
        or path.name.endswith("_tests.rs")
    )

    for line_index in mutation_lines(lines):
        line_number = line_index + 1
        nearby = preceding_comments(lines, line_index)
        function = nearest_function(lines, line_index)
        function_name = function[1] if function else "<module>"

        if is_build_script:
            if "SAFETY:" not in nearby or "build script" not in nearby.lower():
                errors.append(
                    f"{relative_path}:{line_number} ({function_name}): "
                    "build-script environment mutation needs a build-script SAFETY comment"
                )
            if TODO not in nearby:
                errors.append(
                    f"{relative_path}:{line_number} ({function_name}): "
                    "build-script mutation lost its explicit audit TODO"
                )
            continue

        if is_deferred:
            # Keep an explicit marker at every unresolved runtime mutation. A
            # future audit can remove it only after establishing an ordering
            # guarantee or eliminating the process-global mutation.
            if TODO not in nearby:
                errors.append(
                    f"{relative_path}:{line_number} ({function_name}): "
                    "deferred runtime mutation lost its audit TODO"
                )
            continue

        if not has_test_module:
            errors.append(
                f"{relative_path}:{line_number} ({function_name}): "
                "audited mutation is outside a recognized test module"
            )
            continue

        if "SAFETY:" not in nearby:
            errors.append(
                f"{relative_path}:{line_number} ({function_name}): "
                "test environment mutation needs a SAFETY comment"
            )
        if not test_contract(lines, line_index, function):
            errors.append(
                f"{relative_path}:{line_number} ({function_name}): "
                "test environment mutation is not covered by #[serial]"
            )

    # Test/build call sites have all been audited; no old placeholder should
    # remain there. Deferred runtime sites are checked above instead.
    if not is_deferred and not is_build_script:
        for index, line in enumerate(lines):
            if TODO in line:
                errors.append(
                    f"{relative_path}:{index + 1}: stale environment audit TODO remains"
                )
    return errors


def run(root: Path, files: tuple[str, ...]) -> int:
    errors: list[str] = []
    mutation_count = 0
    for relative_path in files:
        path = root / relative_path
        if path.is_file():
            mutation_count += len(mutation_lines(path.read_text(encoding="utf-8").splitlines()))
        errors.extend(check_file(root, relative_path))

    if errors:
        print("environment mutation contract violations:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"environment mutation contract: checked {len(files)} audited files and "
        f"{mutation_count} mutation sites; unresolved runtime sites remain explicit"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the checkout containing this script)",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="audit one relative source file (repeatable; defaults to the full census)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = tuple(args.files) if args.files else AUDITED_FILES
    raise SystemExit(run(args.root.resolve(), selected))
